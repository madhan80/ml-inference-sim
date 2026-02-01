from ml_inference_sim.simulator import Simulator, Cluster, Request
import random
import math

class CapacityFinder:
    def __init__(self, num_devices, ttft_ms, output_tokens_per_sec, input_tokens, output_tokens):
        self.num_devices = num_devices
        self.ttft_ms = ttft_ms
        self.output_tokens_per_sec = output_tokens_per_sec
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def generate_workload(self, rpm: float, duration_min: float) -> list[Request]:
        # Poisson arrival process
        requests = []
        num_requests = int(rpm * duration_min)
        if num_requests == 0:
            return []
            
        # Inter-arrival time = 1 / (requests per second)
        rps = rpm / 60.0
        current_time = 0.0
        
        for i in range(num_requests):
            # Exponential inter-arrival for Poisson process
            inter_arrival = random.expovariate(rps)
            current_time += inter_arrival
            
            req = Request(
                id=i,
                arrival_time=current_time,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens
            )
            requests.append(req)
            
        return requests

    def run_simulation(self, rpm: float, duration_min: float = 1.0):
        cluster = Cluster(self.num_devices, self.ttft_ms, self.output_tokens_per_sec)
        sim = Simulator(cluster)
        
        requests = self.generate_workload(rpm, duration_min)
        sim.run(requests)
        return sim.get_stats()

    def find_max_rpm(self, max_latency_threshold_sec: float = 5.0, duration_min: float = 2.0):
        # Binary search or Step-up approach
        # Theoretical max per device:
        # service_time = ttft + decode_time
        # max_rps_per_device = 1 / service_time
        # theoretical_cluster_rpm = num_devices * max_rps_per_device * 60
        
        decode_time = self.output_tokens / self.output_tokens_per_sec
        service_time = (self.ttft_ms / 1000.0) + decode_time
        theoretical_max_rpm = (self.num_devices / service_time) * 60
        
        print(f"Theoretical Max RPM: {theoretical_max_rpm:.2f}")
        
        low = 1.0
        high = theoretical_max_rpm * 1.5 # Go a bit higher to fail
        best_rpm = 0.0
        
        # Binary search
        for _ in range(10): # Precision
            mid_rpm = (low + high) / 2
            stats = self.run_simulation(mid_rpm, duration_min)
            
            # Criteria for stability:
            # 1. Throughput is close to input RPM (within 5%)
            # 2. Avg Latency is acceptable (not exploding)
            # 3. Queue isn't growing indefinitely (though sim is finite)
            # A simple heuristic: if avg latency < service_time * 5 (allow some queueing)
            
            realized_rpm = stats['throughput_rpm']
            avg_latency = stats['avg_latency_sec']
            
            # Check if system kept up
            # If we asked for X and got < 0.9X, we are saturated (or startup transient, but mostly saturated)
            if realized_rpm < mid_rpm * 0.95 or avg_latency > max_latency_threshold_sec:
                high = mid_rpm
            else:
                best_rpm = mid_rpm
                low = mid_rpm
                
        return best_rpm

if __name__ == "__main__":
    print("Finding Capacity...")
    finder = CapacityFinder(
        num_devices=8,
        ttft_ms=100,
        output_tokens_per_sec=50,
        input_tokens=128,
        output_tokens=128
    ) # service time = 0.1 + 2.56 = 2.66s. 8 devices. Max RPS ~3. Max RPM ~180.
    
    max_rpm = finder.find_max_rpm()
    print(f"Sustainable RPM: {max_rpm:.2f}")
