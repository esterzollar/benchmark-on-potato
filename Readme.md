

# 10 TPS on a "Potato" i3: The 8th Gen Intel Benchmark
**Don't listen to the "experts." Don't ask ChatGPT if it's possible. Just run it.**

This repository proves that you don't need a $3,000 rig or the latest NVIDIA GPU to run high-level local AI. With the right architecture (**MoE**) and the right backend (**OpenVINO**), a 2018 office laptop can outperform expectations.

## 📊 The Findings
Tested on an **HP ProBook 650 G5** (Intel i3-8th Gen, 16GB Dual-Channel RAM) using **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M**.

| Hardware Target | Performance Rating | Avg Speed (TPS) | Character/Vibe |
| :--- | :--- | :--- | :--- |
| **CPU ** | **8.5 / 10** | **8.59 t/s** | Fast out of the gate, snappy logic. |
| **iGPU ** | **9.0 / 10** | **8.99 t/s** | Slow to wake up, but a parallel beast once cruising. |

### Why it works (The MoE Secret)
People will tell you a 16B model is "too big" for an i3. **They are wrong.**
Because this is a **Mixture-of-Experts (MoE)** model, it only activates ~2.4B parameters per token. I tried dense models like **Phi-4-Mini (3.8B)** and **Qwen2.5-3B-Instruct**, and they only hit **2-4 t/s** because they force the CPU to do "brute force" math. The 16B MoE "cheats" by being smarter and faster at the same time.




##  How to Reproduce (The "Potato" Setup)
1. **OS:** Use **Ubuntu**. Windows background bloat will kill your performance.
2. **RAM:** You **MUST** have Dual-Channel (e.g., 2x8GB). Single-channel will cut your speed in half.
3. **Backend:** Install `llama-cpp-python` with the **OpenVINO** backend.
   - *Pro Tip:* Don't try to use OpenVINO standalone unless you enjoy "Dependency Hell." Use the llama-cpp integration.
4. **Mode:** Set your laptop to **Performance Mode** and keep the charger plugged in.

###  Running the Benchmark
Run the provided `deep.py and more accurate deep_decode.py` script to test your own device:
```bash
python3 deep.py
```

---

## Known Weaknesses

* **iGPU Wake-up Call:** The iGPU takes significantly longer to compile the first time (Shader compilation). It might look like it's stuck—don't panic. It's just the "GPU" having his coffee before he starts teaching.
* **Language Drift:** On the iGPU, DeepSeek occasionally hallucinates Chinese characters (it’s a Chinese-base model). The **logic** remains 100% solid, but it might forget it's speaking English for a second.
* **Reading Speed:** While not as fast as a $40/mo cloud subscription, **10 t/s** is faster than the average human can read (5-6 t/s). Why pay for speed you can't even use?

## Stop Asking Chatgpt

I had asked OpenAI's ChatGPT if this was possible, it told me it's "impossible" or "not recommended." 

Stop asking AI and start testing your hardware.

**Check the `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M_result.txt and test2output.txt` for my full TXT logs of Testing!**


> **Note from the Author:**
> I am a developer from Burma. Because high-end GPUs are out of reach for many of us, I spent a month optimizing and "squeezing" budget hardware to see what is actually possible. This project is the result of that effort—proving that accessibility matters more than raw specs.

