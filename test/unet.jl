using UNet: minsize, outputsize

# Generate a fake batch of two images and return it with the expected ouput size.
function dummy_data(;inchannels,
    nclasses = 1,
    volume = false,
    padding = false,
    nlevels = 1)
    sz = minsize(padding, nlevels)
    is = volume ? (sz, sz, sz) : (sz, sz)
    os = UNet.outputsize(is, padding, nlevels)
    rand(Float32, (is..., inchannels, 2)), (os..., nclasses, 2)
end

@testset "unet" begin
    for ic ∈ [1 3],
        nc ∈ [1 5],
        vl ∈ [true false],
        bn ∈ [true false], 
        pg ∈ [true false],
        up ∈ [:convt :nearest :bilinear]

        if vl && up == :bilinear
            continue
        end
        model = unet(inchannels = ic, nclasses = nc, volume = vl, basewidth = 4,
            batchnorm = bn, padding = pg, upsample = up, nlevels = 2)
        x, ys = dummy_data(inchannels = ic, nclasses = nc, volume = vl,
            padding = pg, nlevels = 2)
        @test (model(x) |> size) == ys
    end

    @test_throws ArgumentError unet(inchannels = 3, volume = true,
                                    upsample = :bilinear)
end