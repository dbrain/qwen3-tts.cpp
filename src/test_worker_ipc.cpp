// test_worker_ipc — smoke test for the worker-isolation P1 scaffolding.
//
// Modes:
//   ./test_worker_ipc                  → parent: spawn child via
//                                        spawn_worker(argv[0], …),
//                                        send 1000 PINGs, measure RTT,
//                                        send SHUTDOWN, wait for exit.
//   ./test_worker_ipc --worker <fd>    → child: read frames, echo PONGs
//                                        for PINGs, exit on SHUTDOWN.
//
// Pass: PONG round-trip works, p99 RTT < 200 µs, child exits cleanly.

#include "worker_ipc.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace qwen3_tts;

static int run_worker(int fd) {
    fprintf(stderr, "[worker pid=%d fd=%d] alive\n", (int) getpid(), fd);

    // HELLO
    std::string hello = std::string("{\"pid\":") + std::to_string(getpid())
                      + std::string(",\"build\":\"test_worker_ipc\"}");
    if (send_frame(fd, WorkerFrame::HELLO, 0, hello) != IpcError::OK) {
        fprintf(stderr, "[worker] HELLO failed\n");
        return 1;
    }

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) {
            fprintf(stderr, "[worker] parent EOF, exiting\n");
            return 0;
        }
        if (e != IpcError::OK) {
            fprintf(stderr, "[worker] recv_frame: %s\n", ipc_error_str(e));
            return 2;
        }

        switch (static_cast<WorkerFrame>(hdr.type)) {
            case WorkerFrame::PING:
                // echo back as PONG with the same payload
                if (send_frame(fd, WorkerFrame::PONG, hdr.req_id,
                               payload) != IpcError::OK) {
                    fprintf(stderr, "[worker] PONG send failed\n");
                    return 3;
                }
                break;
            case WorkerFrame::SHUTDOWN:
                fprintf(stderr, "[worker] SHUTDOWN received, exiting\n");
                return 0;
            default:
                fprintf(stderr, "[worker] unexpected frame type=0x%x\n", hdr.type);
                break;
        }
    }
}

static int run_parent(const char * self_argv0) {
    int fd = -1;
    pid_t child = spawn_worker(self_argv0, /*extra_argv=*/{}, &fd);
    if (child < 0) {
        fprintf(stderr, "[parent] spawn_worker failed\n");
        return 10;
    }
    fprintf(stderr, "[parent pid=%d] spawned child pid=%d fd=%d\n",
            (int) getpid(), (int) child, fd);

    // Expect HELLO
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WorkerFrame::HELLO)) {
        fprintf(stderr, "[parent] expected HELLO, got type=0x%x err=%s\n",
                hdr.type, ipc_error_str(e));
        return 11;
    }
    fprintf(stderr, "[parent] HELLO: %.*s\n",
            (int) payload.size(), (const char *) payload.data());

    // PING/PONG bench
    const int N = 1000;
    std::vector<int64_t> rtts; rtts.reserve(N);
    const std::string ping_payload = "{\"k\":1}";
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (send_frame(fd, WorkerFrame::PING, (uint32_t) i + 1,
                       ping_payload) != IpcError::OK) {
            fprintf(stderr, "[parent] PING send failed at i=%d\n", i);
            return 12;
        }
        FrameHeader rh{};
        std::vector<uint8_t> rp;
        e = recv_frame(fd, &rh, &rp);
        if (e != IpcError::OK) {
            fprintf(stderr, "[parent] PONG recv failed at i=%d: %s\n",
                    i, ipc_error_str(e));
            return 13;
        }
        if (rh.type != static_cast<uint32_t>(WorkerFrame::PONG)) {
            fprintf(stderr, "[parent] expected PONG got 0x%x\n", rh.type);
            return 14;
        }
        if (rh.req_id != (uint32_t) i + 1) {
            fprintf(stderr, "[parent] req_id mismatch at i=%d (got %u)\n",
                    i, rh.req_id);
            return 15;
        }
        auto t1 = std::chrono::steady_clock::now();
        rtts.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    std::sort(rtts.begin(), rtts.end());
    auto pct = [&](double p) {
        return rtts[(size_t)(p * (rtts.size() - 1))] / 1000.0;
    };
    double avg = 0;
    for (auto v : rtts) avg += v;
    avg /= rtts.size() * 1000.0;
    printf("RTT µs: avg=%.1f p50=%.1f p90=%.1f p99=%.1f p999=%.1f max=%.1f\n",
           avg, pct(0.50), pct(0.90), pct(0.99), pct(0.999), pct(0.9999));

    // Audio payload round-trip — exercise pack/unpack helpers
    {
        const size_t n_samples = 1024;
        std::vector<float> samples(n_samples);
        for (size_t i = 0; i < n_samples; ++i) samples[i] = (float) i / 32768.0f;
        std::string meta = "{\"sample_rate\":48000,\"n\":" + std::to_string(n_samples) + "}";
        auto packed = pack_audio_payload(meta, samples.data(), n_samples);

        if (send_frame(fd, WorkerFrame::PING, /*req_id=*/9999, packed) != IpcError::OK) {
            fprintf(stderr, "[parent] audio packed send failed\n");
            return 16;
        }
        FrameHeader rh{};
        std::vector<uint8_t> rp;
        if (recv_frame(fd, &rh, &rp) != IpcError::OK) {
            fprintf(stderr, "[parent] audio packed PONG failed\n");
            return 17;
        }
        std::string rmeta;
        std::vector<float> rsamples;
        if (!unpack_audio_payload(rp, &rmeta, &rsamples)) {
            fprintf(stderr, "[parent] unpack_audio_payload failed\n");
            return 18;
        }
        if (rmeta != meta || rsamples.size() != n_samples) {
            fprintf(stderr, "[parent] audio round-trip mismatch (meta='%s' n=%zu)\n",
                    rmeta.c_str(), rsamples.size());
            return 19;
        }
        for (size_t i = 0; i < n_samples; ++i) {
            if (rsamples[i] != samples[i]) {
                fprintf(stderr, "[parent] sample %zu mismatch (%f vs %f)\n",
                        i, rsamples[i], samples[i]);
                return 20;
            }
        }
        printf("audio packed round-trip OK (%zu samples + %zu B meta)\n",
               n_samples, meta.size());
    }

    // SHUTDOWN
    if (send_frame(fd, WorkerFrame::SHUTDOWN, 0, nullptr, 0) != IpcError::OK) {
        fprintf(stderr, "[parent] SHUTDOWN send failed\n");
        return 21;
    }
    int wstat = 0;
    if (waitpid(child, &wstat, 0) < 0) {
        fprintf(stderr, "[parent] waitpid failed: %s\n", strerror(errno));
        return 22;
    }
    if (!WIFEXITED(wstat) || WEXITSTATUS(wstat) != 0) {
        fprintf(stderr, "[parent] child abnormal exit: status=0x%x\n", wstat);
        return 23;
    }
    printf("child exited cleanly\n");
    return 0;
}

int main(int argc, char ** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc >= 3 && std::string(argv[1]) == "--worker") {
        int fd = std::atoi(argv[2]);
        return run_worker(fd);
    }
    return run_parent(argv[0]);
}
