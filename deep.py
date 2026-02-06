from llama_cpp import Llama
import time
import gc
import sys

# Configuration
MODEL_PATH = "path/to/etc/Models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"

# 10 Unique Questions for the CPU 
CPU_QUESTIONS = [
    "Logic: If all A are B and some B are C, is it necessarily true that some A are C? Explain your reasoning step-by-step.",
    "Health: What are the primary lifestyle factors that contribute to improving heart health, and how do they interact?",
    "History: Summarize the significance of the Magna Carta in the context of modern constitutional law.",
    "Coding: Write a Python function using recursion to solve the Tower of Hanoi problem.",
    "Creativity: Write a 10-line poem about the intersection of ancient nature and cyberpunk technology.",
    "Fact: Provide a detailed biography of Marie Curie, focusing on her two Nobel Prizes.",
    "Math: Explain the concept of the 'Golden Ratio' and find three examples of it in nature.",
    "Tech: Explain how a 'Virtual Private Network' (VPN) works using an analogy involving a physical envelope.",
    "Ethics: Discuss the ethical dilemmas of self-driving cars in 'trolley problem' scenarios.",
    "Food: Explain the science of 'Maillard Reaction' and why it is crucial for searing a steak."
]

# 10 Different Unique Questions for the iGPU 
GPU_QUESTIONS = [
    "Logic: Brothers and sisters I have none, but that man's father is my father's son. Who is in the photo? Walk me through the logic.",
    "Health: Explain the physiological differences between Type 1 and Type 2 diabetes for a medical student.",
    "History: Analyze the socio-economic causes that led directly to the storming of the Bastille.",
    "Coding: Write a Python script that implements a basic Blockchain with blocks, hashes, and a simple proof-of-work.",
    "Creativity: Describe a futuristic library where books are stored as biological DNA memories in 100 words.",
    "Fact: Explain the 'Tallest Mountain' debate: Everest vs. Mauna Kea vs. Chimborazo.",
    "Math: What is the 'Monty Hall Problem'? Explain why the math suggests you should always switch doors.",
    "Tech: Compare HDD, SSD, and NVMe technologies in terms of latency, IOPS, and physical longevity.",
    "Ethics: Should AI models be allowed to have 'copyright' over the art or code they generate?",
    "Food: Detail the chemical process of fermentation in sourdough bread and why it differs from commercial yeast."
]

def run_streaming_bench(device_label, questions, n_gpu_layers):
    print(f"\n{'='*70}\n>>> STARTING BATTLE: {device_label}\n{'='*70}")
    
    try:
        # Load model fresh
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,       # Comfortable context for 16GB dual-channel RAM
            n_threads=4,      # Optimized for i3-8th Gen
            n_gpu_layers=n_gpu_layers, 
            verbose=False
        )

        total_tokens = 0
        total_gen_time = 0

        for i, q in enumerate(questions):
            print(f"\n[Q{i+1}/10] {q[:65]}...")
            print(f"Response: ", end="", flush=True)
            
            start_trigger = time.perf_counter()
            q_token_count = 0
            
            # stream=True with NO max_tokens limit (None)
            stream = llm(q, max_tokens=None, stream=True, stop=["<|EOT|>"])
            
            for output in stream:
                text = output["choices"][0]["text"]
                print(text, end="", flush=True)
                q_token_count += 1
            
            end_trigger = time.perf_counter()
            duration = end_trigger - start_trigger
            
            # Calculate speed for this specific question
            tps = q_token_count / duration
            print(f"\n\n--- Stats: {q_token_count} tokens | {tps:.2f} t/s ---")
            
            total_tokens += q_token_count
            total_gen_time += duration

        avg_tps = total_tokens / total_gen_time
        print(f"\n[DEVICE FINAL] {device_label}: {avg_tps:.2f} tokens/sec (Average)")
        
        # Explicit Unload to prevent OOM on 16GB
        del llm
        gc.collect()
        time.sleep(5) # Let the hardware cool down
        return avg_tps

    except Exception as e:
        print(f"\nError on {device_label}: {e}")
        return 0

if __name__ == "__main__":
    # Test 1: CPU 
    cpu_score = run_streaming_bench("CPU ", CPU_QUESTIONS, 0)
    
    # Test 2: iGPU 
    gpu_score = run_streaming_bench("iGPU ", GPU_QUESTIONS, 1)

    # Final Comparison Report
    print(f"\n{'#'*60}")
    print(f"{'DEVICE':<25} | {'AVG SPEED (TPS)':<20}")
    print(f"{'-'*55}")
    print(f"{'CPU ':<25} | {cpu_score:<20.2f}")
    print(f"{'iGPU ':<25} | {gpu_score:<20.2f}")
    print(f"{'#'*60}")
    
    if gpu_score > cpu_score:
        diff = ((gpu_score / cpu_score) - 1) * 100
        print(f"WINNER: iGPU is {diff:.1f}% faster over 10 unique tasks.")
    else:
        diff = ((cpu_score / gpu_score) - 1) * 100 if gpu_score > 0 else 0
        print(f"WINNER: CPU is {diff:.1f}% faster over 10 unique tasks.")
