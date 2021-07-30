using UNet
using Test
using CUDA
if CUDA.has_cuda_gpu()
    CUDA.allowscalar(false)
    device = gpu
else
    device = cpu
end

@testset "UNet.jl" begin
    model = unet(inchannels = 3) |> device
    x = rand(572, 572, 3, 1) |> device
    @test (model(x) |> device) == (388, 388, 1, 1)
end