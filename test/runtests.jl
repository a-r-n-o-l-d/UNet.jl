using Test
using UNet
#=using Flux # trop long à virer
using CUDA # trop long à virer
if CUDA.has_cuda_gpu()
    CUDA.allowscalar(false)
    device = gpu
else
    device = cpu
end
=#
@testset "UNet.jl" begin
    a = rand(392, 392, 64, 1)
    b = rand(568, 568, 64, 1)
    c = UNet.CropCat((88, 88, 0, 0))
    @test (c(a, b) |> size) == (392, 392, 128, 1)

    @test UNet.utrim(1) == 88

    # trop long à virer
    #model = unet(inchannels = 3) |> device
    #x = rand(572, 572, 3, 1) |> device
    #@test (model(x) |> device) == (388, 388, 1, 1)
end