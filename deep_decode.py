from llama_cpp import Llama
import time, gc
from statistics import mean
MODEL_PATH = "/home/esterzollar/jarvis_py/Models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"

N_CTX = 4096
N_THREADS = 4
STOP = ["<|EOT|>"]
MAX_TOKENS = 256 
TEMPERATURE = 0.2
TOP_P = 0.9
QUESTIONS = [
    "Logic: If all A are B and some B are C, is it necessarily true that some A are C? Explain your reasoning step-by-step.",
    "Health: What are the primary lifestyle factors that contribute to improving heart health, and how do they interact?",
    "History: Summarize the significance of the Magna Carta in the context of modern constitutional law.",
    "Coding: Write a Python function using recursion to solve the Tower of Hanoi problem.",
    "Creativity: Write a 10-line poem about the intersection of ancient nature and cyberpunk technology.",
    "Fact: Provide a detailed biography of Marie Curie, focusing on her two Nobel Prizes.",
    "Math: Explain the concept of the 'Golden Ratio' and find three examples of it in nature.",
    "Tech: Explain how a 'Virtual Private Network' (VPN) works using an analogy involving a physical envelope.",
    "Ethics: Discuss the ethical dilemmas of self-driving cars in 'trolley problem' scenarios.",
    "Food: Explain the science of 'Maillard Reaction' and why it is crucial for searing a steak.",
]

def tok_count(llm: Llama, s: str) -> int:
    return len(llm.tokenize(s.encode("utf-8")))

def unload(llm: Llama):
    del llm
    gc.collect()
    time.sleep(2)

def run_bench(device_label: str, n_gpu_layers: int) -> dict:
    print(f"\n{'='*70}\n>>> STARTING BENCH: {device_label}\n{'='*70}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    _ = llm("Warmup: say 'ok'.", max_tokens=8, temperature=0, top_p=1.0, stop=STOP)

    per_q = []
    decode_tps_list = []
    ttft_list = []

    try:
        for i, q in enumerate(QUESTIONS, start=1):
            prompt_tokens = tok_count(llm, q)

            print(f"\n[Q{i}/{len(QUESTIONS)}] {q[:70]}...")
            print("Response: ", end="", flush=True)

            start = time.perf_counter()
            first = None
            out_text = ""

            stream = llm(
                q,
                max_tokens=MAX_TOKENS,
                stream=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=STOP,
            )

            for chunk in stream:
                if first is None:
                    first = time.perf_counter()
                text = chunk["choices"][0]["text"]
                out_text += text
                print(text, end="", flush=True)

            end = time.perf_counter()
            first = first or end  # safety if no output

            completion_tokens = tok_count(llm, out_text)
            total_s = end - start
            ttft = first - start
            decode_s = max(1e-9, end - first)
            decode_tps = completion_tokens / decode_s

            per_q.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_s": total_s,
                "ttft_s": ttft,
                "decode_tps": decode_tps,
            })

            decode_tps_list.append(decode_tps)
            ttft_list.append(ttft)

            print(
                f"\n--- Stats: prompt={prompt_tokens} tok | "
                f"completion={completion_tokens} tok | "
                f"TTFT={ttft:.2f}s | decode={decode_tps:.2f} tok/s | total={total_s:.2f}s ---"
            )

    except KeyboardInterrupt:
        print("\n[Interrupted] Returning partial results...")

    summary = {
        "device": device_label,
        "avg_decode_tps": mean(decode_tps_list) if decode_tps_list else 0.0,
        "avg_ttft_s": mean(ttft_list) if ttft_list else 0.0,
        "samples": len(decode_tps_list),
    }

    print(
        f"\n[DEVICE FINAL] {device_label}: "
        f"avg decode={summary['avg_decode_tps']:.2f} tok/s | "
        f"avg TTFT={summary['avg_ttft_s']:.2f}s | "
        f"samples={summary['samples']}"
    )

    unload(llm)
    return summary

if __name__ == "__main__":
    cpu = run_bench("CPU (n_gpu_layers=0)", n_gpu_layers=0)
    igpu = run_bench("iGPU (n_gpu_layers=1)", n_gpu_layers=1)

    print(f"\n{'#'*60}")
    print(f"{'DEVICE':<28} | {'AVG DECODE TPS':<14} | {'AVG TTFT (s)':<12} | {'N':<3}")
    print(f"{'-'*60}")
    print(f"{cpu['device']:<28} | {cpu['avg_decode_tps']:<14.2f} | {cpu['avg_ttft_s']:<12.2f} | {cpu['samples']:<3}")
    print(f"{igpu['device']:<28} | {igpu['avg_decode_tps']:<14.2f} | {igpu['avg_ttft_s']:<12.2f} | {igpu['samples']:<3}")
    print(f"{'#'*60}")

    if cpu["avg_decode_tps"] > 0 and igpu["avg_decode_tps"] > 0:
        diff = (igpu["avg_decode_tps"] / cpu["avg_decode_tps"] - 1.0) * 100.0
        winner = "iGPU" if diff > 0 else "CPU"
        print(f"WINNER: {winner} by {abs(diff):.1f}% (decode TPS, fixed max_tokens={MAX_TOKENS})")

