using Test
using Flux
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
    sz = UNet.minsize(padding, nlevels)
    is = volume ? (sz, sz, sz) : (sz, sz)
    os = UNet.outputsize(is, padding, nlevels)
    rand(Float32, (is..., inchannels, 2)), (os..., nclasses, 2)
end

include("uchain.jl")
include("unet.jl")