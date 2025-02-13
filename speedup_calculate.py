import matplotlib.pyplot as plt


sequential_time = float(136.786)
openmp_time = float(59.0669)
cuda_time = float(2.67418)


speedup_openmp = sequential_time / openmp_time
speedup_cuda = sequential_time / cuda_time


print("\n  Speedup Results:")
print(f"OpenMP Speedup: {speedup_openmp:.2f}x")
print(f"CUDA Speedup: {speedup_cuda:.2f}x")


implementations = ['Sequential', 'OpenMP', 'CUDA']
execution_times = [sequential_time, openmp_time, cuda_time]

plt.figure(figsize=(8,5))
plt.bar(implementations, execution_times, color=['red', 'blue', 'green'])
plt.ylabel("Execution Time (seconds)")
plt.title("Performance Comparison: Sequential vs OpenMP vs CUDA")
plt.ylim(0, max(execution_times) * 1.2) 
plt.savefig("execution_time_comparison.png", dpi=300)  
plt.close()


speedups = [1.0, speedup_openmp, speedup_cuda]

plt.figure(figsize=(8,5))
plt.bar(implementations, speedups, color=['gray', 'blue', 'green'])
plt.ylabel("Speedup Factor")
plt.title("Speedup Comparison: OpenMP & CUDA vs Sequential")
plt.ylim(0, max(speedups) * 1.2)
plt.savefig("speedup_comparison.png", dpi=300)  
plt.close()


plt.figure(figsize=(8,5))
plt.plot(implementations, speedups, marker='o', linestyle='-', color='black', label="Speedup Factor")
plt.axhline(y=1.0, color='r', linestyle='--', label="Baseline (Sequential)")
plt.ylabel("Speedup Factor")
plt.title("Scaling Efficiency of OpenMP & CUDA")
plt.legend()
plt.grid(True)
plt.savefig("speedup_scaling.png", dpi=300)  
plt.close()

print("\n Graphs saved as PNG files:")
print("  - execution_time_comparison.png")
print("  - speedup_comparison.png")
print("  - speedup_scaling.png")
