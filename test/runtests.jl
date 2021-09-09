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

# Generate a fake batch of two images and return it with the expected ouput size.
function dummy_data(;inchannels,
                     nclasses = 1,
                     volume = false,
                     padding = false,
                     nlevels = 1)
#=    is = volume ? (4, 4, 4) : (4, 4)              # image size
    ip, op = UNet.padding(is, padding = false, nlevels = nlevels)                # image padding
    inp = ((@. is + 2 * ip[1])..., inchannels, 2) # input batch size
    oup = ((@. is + 2 * op[1])..., nclasses, 2)   # output size
    =#
    sz = UNet.minsize(padding, nlevels)
    is = volume ? (sz, sz, sz) : (sz, sz)
    os = UNet.outputsize(is, padding, nlevels)
    rand(Float32, (is..., inchannels, 2)), (os..., nclasses, 2)
end

@testset "UNet.jl" begin

    for ic ∈ [1 3], nc ∈ [1 5], vl ∈ [true false], bn ∈ [true false], 
        pg ∈ [true false], up ∈ [:convt :nearest :bilinear]
        if vl == true && up == :bilinear
            continue
        end
        model = unet(inchannels = ic, nclasses = nc, volume = vl, base = 4,
            batchnorm = bn, padding = pg, upsample = up, nlevels = 2)
        x, oup = dummy_data(inchannels = ic, nclasses = nc, volume = vl,
            padding = pg, nlevels = 2)
        @test (model(x) |> size) == oup
    end

    #=a = rand(392, 392, 64, 1)
    b = rand(568, 568, 64, 1)
    c = UNet.CropCat((88, 88, 0, 0))
    @test (c(a, b) |> size) == (392, 392, 128, 1)

    @test UNet.utrim(1) == 88=#

    # trop long à virer
    #model = unet(inchannels = 3) |> device
    #x = rand(572, 572, 3, 1) |> device
    #@test (model(x) |> device) == (388, 388, 1, 1)
end