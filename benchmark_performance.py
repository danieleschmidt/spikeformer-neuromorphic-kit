#!/usr/bin/env python3
"""Performance benchmarking without heavy dependencies."""

import time
import os
import sys
import gc
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import traceback
from typing import Dict, Any, List, Tuple

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        info = {
            'cpu_count': os.cpu_count(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if HAS_PSUTIL:
            info.update({
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            })
        else:
            info.update({
                'memory_total_gb': 'Unknown (psutil not available)',
                'memory_available_gb': 'Unknown (psutil not available)'
            })
        
        return info
    except Exception as e:
        return {'error': str(e)}

def benchmark_file_operations() -> Dict[str, float]:
    """Benchmark file I/O operations."""
    results = {}
    
    # Test file reading
    test_files = [
        'spikeformer/__init__.py',
        'spikeformer/models.py',
        'README.md',
        'pyproject.toml'
    ]
    
    total_read_time = 0
    total_size = 0
    
    for file_path in test_files:
        if os.path.exists(file_path):
            start_time = time.time()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_size += len(content)
                end_time = time.time()
                total_read_time += (end_time - start_time)
            except Exception:
                pass
    
    results['file_read_speed_mb_per_sec'] = (total_size / (1024 * 1024)) / max(total_read_time, 0.001)
    results['total_read_time'] = total_read_time
    
    return results

def benchmark_string_operations() -> Dict[str, float]:
    """Benchmark string operations."""
    results = {}
    
    # String concatenation
    start_time = time.time()
    result_str = ""
    for i in range(10000):
        result_str += f"test_string_{i}_"
    end_time = time.time()
    results['string_concat_time'] = end_time - start_time
    
    # String formatting
    start_time = time.time()
    formatted_strings = []
    for i in range(10000):
        formatted = f"Formatted string {i} with value {i*2.5:.2f}"
        formatted_strings.append(formatted)
    end_time = time.time()
    results['string_format_time'] = end_time - start_time
    
    # String parsing
    test_data = "field1,field2,field3,field4\n" * 1000
    start_time = time.time()
    lines = test_data.split('\n')
    for line in lines:
        if line:
            fields = line.split(',')
    end_time = time.time()
    results['string_parse_time'] = end_time - start_time
    
    return results

def benchmark_data_structures() -> Dict[str, float]:
    """Benchmark basic data structure operations."""
    results = {}
    
    # List operations
    start_time = time.time()
    test_list = []
    for i in range(100000):
        test_list.append(i)
    
    # List search
    for i in range(0, 100000, 1000):
        _ = i in test_list
    
    end_time = time.time()
    results['list_operations_time'] = end_time - start_time
    
    # Dictionary operations
    start_time = time.time()
    test_dict = {}
    for i in range(100000):
        test_dict[f"key_{i}"] = i * 2
    
    # Dictionary lookup
    for i in range(0, 100000, 1000):
        _ = test_dict.get(f"key_{i}")
    
    end_time = time.time()
    results['dict_operations_time'] = end_time - start_time
    
    return results

def benchmark_computation() -> Dict[str, float]:
    """Benchmark computational operations."""
    results = {}
    
    # Mathematical operations
    start_time = time.time()
    total = 0
    for i in range(1000000):
        total += i * 2.5 + 1.0 / max(i + 1, 1)
    end_time = time.time()
    results['math_operations_time'] = end_time - start_time
    
    # Matrix-like operations (without numpy)
    start_time = time.time()
    matrix_a = [[i + j for j in range(100)] for i in range(100)]
    matrix_b = [[i * j for j in range(100)] for i in range(100)]
    
    # Simple matrix multiplication
    result_matrix = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_b[0])):
            sum_val = 0
            for k in range(len(matrix_b)):
                sum_val += matrix_a[i][k] * matrix_b[k][j]
            row.append(sum_val)
        result_matrix.append(row)
    
    end_time = time.time()
    results['matrix_operations_time'] = end_time - start_time
    
    return results

def benchmark_memory() -> Dict[str, float]:
    """Benchmark memory operations."""
    results = {}
    
    if HAS_PSUTIL:
        # Initial memory
        initial_memory = psutil.virtual_memory().available
    
    # Memory allocation
    start_time = time.time()
    large_lists = []
    for i in range(10):
        large_list = list(range(100000))
        large_lists.append(large_list)
    end_time = time.time()
    
    results['memory_alloc_time'] = end_time - start_time
    
    if HAS_PSUTIL:
        # Memory after allocation
        after_alloc_memory = psutil.virtual_memory().available
        results['memory_used_mb'] = (initial_memory - after_alloc_memory) / (1024 * 1024)
    else:
        results['memory_used_mb'] = 'Unknown (psutil not available)'
    
    # Cleanup and measure garbage collection
    start_time = time.time()
    del large_lists
    gc.collect()
    end_time = time.time()
    
    results['gc_time'] = end_time - start_time
    
    return results

def run_benchmark_suite() -> Dict[str, Any]:
    """Run complete benchmark suite."""
    print("🏁 Starting Performance Benchmark Suite")
    print("=" * 60)
    
    system_info = get_system_info()
    print(f"💻 System Info:")
    print(f"   CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
    memory_total = system_info.get('memory_total_gb', 'Unknown')
    memory_available = system_info.get('memory_available_gb', 'Unknown')
    if isinstance(memory_total, (int, float)) and isinstance(memory_available, (int, float)):
        print(f"   Memory: {memory_total:.1f} GB total, {memory_available:.1f} GB available")
    else:
        print(f"   Memory: {memory_total} total, {memory_available} available")
    print(f"   Platform: {system_info.get('platform', 'Unknown')}")
    
    benchmarks = [
        ("File Operations", benchmark_file_operations),
        ("String Operations", benchmark_string_operations),
        ("Data Structures", benchmark_data_structures),
        ("Computation", benchmark_computation),
        ("Memory Operations", benchmark_memory)
    ]
    
    all_results = {'system_info': system_info}
    total_time = 0
    
    for benchmark_name, benchmark_func in benchmarks:
        print(f"\n📊 Running {benchmark_name} Benchmark...")
        print("-" * 40)
        
        try:
            start_time = time.time()
            results = benchmark_func()
            end_time = time.time()
            
            benchmark_time = end_time - start_time
            total_time += benchmark_time
            
            all_results[benchmark_name] = results
            all_results[benchmark_name]['total_benchmark_time'] = benchmark_time
            
            # Print results
            for metric, value in results.items():
                if isinstance(value, float):
                    if 'time' in metric:
                        print(f"   {metric}: {value:.4f} seconds")
                    elif 'mb' in metric or 'gb' in metric:
                        print(f"   {metric}: {value:.2f}")
                    else:
                        print(f"   {metric}: {value:.2f}")
                else:
                    print(f"   {metric}: {value}")
            
            print(f"   ✅ Completed in {benchmark_time:.4f} seconds")
            
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            traceback.print_exc()
            all_results[benchmark_name] = {'error': str(e)}
    
    # Performance score (lower is better)
    performance_score = 0
    weights = {
        'File Operations': 1.0,
        'String Operations': 2.0,
        'Data Structures': 2.0,
        'Computation': 3.0,
        'Memory Operations': 1.5
    }
    
    for benchmark_name, weight in weights.items():
        if benchmark_name in all_results and 'total_benchmark_time' in all_results[benchmark_name]:
            performance_score += all_results[benchmark_name]['total_benchmark_time'] * weight
    
    all_results['overall_performance_score'] = performance_score
    all_results['total_benchmark_time'] = total_time
    
    print(f"\n🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"Performance Score: {performance_score:.2f} (lower is better)")
    
    # Performance rating
    if performance_score < 5.0:
        rating = "🚀 Excellent"
    elif performance_score < 10.0:
        rating = "✅ Good"
    elif performance_score < 20.0:
        rating = "⚠️ Average"
    else:
        rating = "🐌 Needs Improvement"
    
    print(f"Performance Rating: {rating}")
    
    return all_results

def main():
    """Main benchmark runner."""
    try:
        results = run_benchmark_suite()
        
        # Save results to file
        import json
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to benchmark_results.json")
        print("\n🎉 Performance benchmark completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)