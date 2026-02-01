import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque

@dataclass
class Request:
    id: int
    arrival_time: float
    input_tokens: int
    output_tokens: int
    start_time: float = -1.0
    completion_time: float = -1.0

    @property
    def latency(self):
        if self.completion_time < 0 or self.arrival_time < 0:
            return 0.0
        return self.completion_time - self.arrival_time

@dataclass(order=True)
class Event:
    timestamp: float
    type: str = field(compare=False)
    request_id: int = field(compare=False)
    device_id: Optional[int] = field(default=None, compare=False)

class Device:
    def __init__(self, id: int, ttft_ms: float, output_tokens_per_sec: float):
        self.id = id
        self.ttft_ms = ttft_ms
        self.output_tokens_per_sec = output_tokens_per_sec
        self.is_busy = False
        self.current_request: Optional[Request] = None
        self.total_busy_time = 0.0

    def calculate_duration(self, req: Request) -> float:
        ttft_sec = self.ttft_ms / 1000.0
        decode_sec = req.output_tokens / self.output_tokens_per_sec
        return ttft_sec + decode_sec

class Cluster:
    def __init__(self, num_devices: int, ttft_ms: float, output_tokens_per_sec: float):
        self.devices = [Device(i, ttft_ms, output_tokens_per_sec) for i in range(num_devices)]
        # Simple Round Robin to start
        self._rr_index = 0

    def get_next_device(self) -> Device:
        # Implementing Least Connections (approximated by availability) or Round Robin?
        # Let's try to find a free device first
        for d in self.devices:
            if not d.is_busy:
                return d
        
        # If all busy, use Round Robin to assign (simulating a queue at the device or dispatcher)
        # Note: In a real system, you might queue at the cluster level. 
        # For this sim, we'll assume a global cluster queue logic in the Simulator,
        # so this method might just be "pick best candidate if one is free".
        return None

class Simulator:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.events = [] # Min-heap for events
        self.current_time = 0.0
        self.waiting_queue: Deque[Request] = deque()
        self.completed_requests: List[Request] = []
        self.requests_map = {}

    def schedule_event(self, timestamp: float, type: str, request_id: int, device_id: Optional[int] = None):
        heapq.heappush(self.events, Event(timestamp, type, request_id, device_id))

    def run(self, requests: List[Request]):
        # Load initial events
        for req in requests:
            self.requests_map[req.id] = req
            self.schedule_event(req.arrival_time, "REQUEST_ARRIVAL", req.id)
        
        while self.events:
            event = heapq.heappop(self.events)
            self.current_time = event.timestamp
            
            if event.type == "REQUEST_ARRIVAL":
                self.handle_arrival(event.request_id)
            elif event.type == "DEVICE_FREE":
                self.handle_completion(event.request_id, event.device_id)

    def handle_arrival(self, request_id: int):
        req = self.requests_map[request_id]
        
        # Try to schedule immediately
        device = self.cluster.get_next_device()
        if device:
            self._start_job(device, req)
        else:
            self.waiting_queue.append(req)

    def handle_completion(self, request_id: int, device_id: int):
        req = self.requests_map[request_id]
        device = self.cluster.devices[device_id]
        
        req.completion_time = self.current_time
        self.completed_requests.append(req)
        
        device.is_busy = False
        device.current_request = None
        
        # Check queue
        if self.waiting_queue:
            next_req = self.waiting_queue.popleft()
            self._start_job(device, next_req)

    def _start_job(self, device: Device, req: Request):
        device.is_busy = True
        device.current_request = req
        req.start_time = self.current_time
        
        duration = device.calculate_duration(req)
        device.total_busy_time += duration
        
        completion_time = self.current_time + duration
        self.schedule_event(completion_time, "DEVICE_FREE", req.id, device.id)

    def get_stats(self):
        if not self.completed_requests:
            return "No requests completed."
        
        latencies = [r.latency for r in self.completed_requests]
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        throughput = len(self.completed_requests) / (self.current_time if self.current_time > 0 else 1) * 60 # RPM
        
        return {
            "total_requests": len(self.completed_requests),
            "avg_latency_sec": avg_lat,
            "max_latency_sec": max_lat,
            "throughput_rpm": throughput,
            "total_time_sec": self.current_time
        }

if __name__ == "__main__":
    # Test Run
    print("Running basic simulation...")
    cluster = Cluster(num_devices=2, ttft_ms=100, output_tokens_per_sec=50)
    sim = Simulator(cluster)
    
    # Generate 10 requests arriving every 0.1s
    test_requests = []
    for i in range(10):
        req = Request(id=i, arrival_time=i*0.1, input_tokens=50, output_tokens=100)
        test_requests.append(req)
        
    sim.run(test_requests)
    stats = sim.get_stats()
    print("Stats:", stats)
