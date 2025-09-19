import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class ParallelPMField(nn.Module):
    """
    Vectorized PMField implementation with temporal parallelism.
    
    This implements the core Pushing-Medium gravitational equations:
    - Refractive index: n(r) = 1 + Σμᵢ/|r-rᵢ|
    - Gradient: ∇ln(n) = ∇n/n = -Σ(μᵢ/|r-rᵢ|³)(r-rᵢ)
    - Flow acceleration: a = -c²∇ln(n)
    
    Enhanced with vectorized center operations and batch processing.
    """
    
    def __init__(self, d_latent=8, n_centers=64, steps=4, dt=0.15, beta=1.2, 
                 clamp=3.0, temporal_parallel=True, chunk_size=16):
        super().__init__()
        # Better initialization for gravitational centers
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent) * 0.8)  # Slightly wider spread
        # Initialize mus with more variation for better specialization
        self.mus = nn.Parameter(torch.ones(n_centers) * 0.5 + torch.randn(n_centers) * 0.1)
        self.steps = steps
        self.dt = dt
        self.beta = beta
        self.clamp = clamp
        self.temporal_parallel = temporal_parallel
        self.chunk_size = chunk_size
        
        # Cache for vectorized operations
        self.register_buffer('_eye', torch.eye(d_latent))
        self.register_buffer('_eps', torch.tensor(1e-4))
        
    def vectorized_grad_ln_n(self, z: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of ∇ln(n) for all centers simultaneously.
        
        Implements: ∇ln(n) = -Σ(μᵢ/|r-rᵢ|³)(r-rᵢ) / n_total
        where n_total = 1 + Σμᵢ/|r-rᵢ|
        """
        B, D = z.shape
        N = self.centers.shape[0]
        
        # Vectorized distance computation: (B, N, D)
        rvec = z.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, N, D)
        r2 = torch.sum(rvec * rvec, dim=2) + self._eps  # (B, N)
        r = torch.sqrt(r2)  # (B, N)
        
        # Vectorized refractive index: n = 1 + Σμᵢ/rᵢ
        n_contributions = self.mus.unsqueeze(0) / r  # (B, N)
        n_total = 1.0 + torch.sum(n_contributions, dim=1, keepdim=True)  # (B, 1)
        
        # Vectorized gradient computation
        r3 = r2 * r  # (B, N)
        grad_prefactor = -self.mus.unsqueeze(0).unsqueeze(2) / r3.unsqueeze(2)  # (B, N, 1)
        grad_contributions = grad_prefactor * rvec  # (B, N, D)
        grad_ln_n = torch.sum(grad_contributions, dim=1) / n_total  # (B, D)
        
        return grad_ln_n
    
    def temporal_pipeline_step(self, z: torch.Tensor) -> torch.Tensor:
        """Single temporal step with vectorized operations."""
        grad = self.vectorized_grad_ln_n(z)
        z_new = z + self.dt * self.beta * grad
        return torch.clamp(z_new, -self.clamp, self.clamp)
    
    def parallel_temporal_evolution(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parallel temporal evolution using pipeline overlapping.
        
        For embarrassingly parallel computation, each PMFlow center
        acts independently like gravitational point masses.
        """
        if not self.temporal_parallel or self.steps <= 2:
            # Standard sequential evolution
            for _ in range(self.steps):
                z = self.temporal_pipeline_step(z)
            return z
        
        # Pipeline parallel evolution
        B = z.shape[0]
        if B <= self.chunk_size:
            # Small batch - use standard evolution
            for _ in range(self.steps):
                z = self.temporal_pipeline_step(z)
            return z
        
        # Large batch - use chunked parallel processing
        chunks = torch.chunk(z, math.ceil(B / self.chunk_size), dim=0)
        evolved_chunks = []
        
        for chunk in chunks:
            z_chunk = chunk
            for _ in range(self.steps):
                z_chunk = self.temporal_pipeline_step(z_chunk)
            evolved_chunks.append(z_chunk)
        
        return torch.cat(evolved_chunks, dim=0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal parallelism."""
        return self.parallel_temporal_evolution(z)

class VectorizedLateralEI(nn.Module):
    """
    Vectorized lateral excitation-inhibition with memory optimization.
    
    Implements biological cortical column lateral interactions with
    efficient memory usage for large batch processing.
    """
    
    def __init__(self, sigma_e=0.6, sigma_i=1.2, k_e=0.8, k_i=1.0, 
                 gain=0.05, chunk_ei=True, chunk_size=32):
        super().__init__()
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.k_e = k_e
        self.k_i = k_i
        self.gain = gain
        self.chunk_ei = chunk_ei
        self.chunk_size = chunk_size
        
        # Pre-compute constants
        self.register_buffer('_sigma_e2', torch.tensor(2 * sigma_e ** 2))
        self.register_buffer('_sigma_i2', torch.tensor(2 * sigma_i ** 2))
        
    def compute_ei_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """Compute excitation-inhibition kernel efficiently."""
        B = z.shape[0]
        
        if not self.chunk_ei or B <= self.chunk_size:
            # Standard computation for small batches
            dist2 = torch.cdist(z, z).pow(2)
            Ke = self.k_e * torch.exp(-dist2 / self._sigma_e2)
            Ki = self.k_i * torch.exp(-dist2 / self._sigma_i2)
            K = Ke - Ki
            return K / (K.sum(1, keepdim=True) + 1e-6)
        
        # Chunked computation for memory efficiency
        K = torch.zeros(B, B, device=z.device, dtype=z.dtype)
        chunk_size = self.chunk_size
        
        for i in range(0, B, chunk_size):
            i_end = min(i + chunk_size, B)
            z_i = z[i:i_end]
            
            for j in range(0, B, chunk_size):
                j_end = min(j + chunk_size, B)
                z_j = z[j:j_end]
                
                dist2 = torch.cdist(z_i, z_j).pow(2)
                Ke = self.k_e * torch.exp(-dist2 / self._sigma_e2)
                Ki = self.k_i * torch.exp(-dist2 / self._sigma_i2)
                K[i:i_end, j:j_end] = Ke - Ki
        
        # Normalize rows
        K = K / (K.sum(1, keepdim=True) + 1e-6)
        return K
    
    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass with vectorized EI computation."""
        with torch.no_grad():
            K = self.compute_ei_kernel(z)
        return self.gain * (K @ h)

class AdaptiveScheduler:
    """
    Adaptive scheduling for temporal parallelism.
    
    Automatically adjusts chunk sizes and parallel strategies
    based on hardware capabilities and batch size.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_fraction = 0.8  # Use 80% of available memory
        self.min_chunk_size = 8
        self.max_chunk_size = 128
        
    def get_optimal_chunk_size(self, batch_size: int, feature_dim: int) -> int:
        """Determine optimal chunk size based on memory constraints."""
        if self.device.type == 'cuda':
            # Estimate memory usage
            memory_per_sample = feature_dim * 4  # float32
            available_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory *= self.memory_fraction
            
            max_samples = int(available_memory / memory_per_sample)
            chunk_size = min(max_samples, batch_size, self.max_chunk_size)
            chunk_size = max(chunk_size, self.min_chunk_size)
        else:
            # CPU - use smaller chunks
            chunk_size = min(batch_size, 32)
        
        return chunk_size
    
    def should_use_temporal_parallel(self, batch_size: int, steps: int) -> bool:
        """Decide whether to use temporal parallelism."""
        return batch_size >= 16 and steps >= 3

@torch.no_grad()
def vectorized_pm_plasticity(pmfield: ParallelPMField, z_batch: torch.Tensor, 
                           h_batch: torch.Tensor, mu_lr=1e-3, c_lr=1e-3):
    """
    Vectorized plasticity update implementing local Hebbian-style learning.
    
    This implements the biological neural adaptation based on activity patterns
    and gravitational center dynamics.
    """
    s2 = 0.8 ** 2
    C = pmfield.centers  # (N, D)
    B, D = z_batch.shape
    N = C.shape[0]
    
    # Vectorized distance computation
    z_expanded = z_batch.unsqueeze(1)  # (B, 1, D)
    C_expanded = C.unsqueeze(0)  # (1, N, D)
    dist2 = torch.sum((C_expanded - z_expanded) ** 2, dim=2)  # (B, N)
    
    # Vectorized weight computation
    W = torch.exp(-dist2 / (2 * s2))  # (B, N)
    
    # Vectorized activity-based updates
    hpow = torch.sum(h_batch * h_batch, dim=1, keepdim=True)  # (B, 1)
    drive = torch.mean(W * hpow, dim=0)  # (N,)
    
    # Update mus (gravitational strengths)
    pmfield.mus.add_(mu_lr * (drive - 0.1 * pmfield.mus))
    
    # Update centers (gravitational positions)
    W_sum = torch.sum(W, dim=0, keepdim=True).T + 1e-6  # (N, 1)
    weighted_z = torch.sum(W.T.unsqueeze(2) * z_batch.unsqueeze(0), dim=1)  # (N, D)
    target = weighted_z / W_sum  # (N, D)
    pmfield.centers.add_(c_lr * (target - C))