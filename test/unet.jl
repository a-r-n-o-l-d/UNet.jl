@testset "unet" begin

    for ic ∈ [1 3],
        nc ∈ [1 5],
        vl ∈ [true false],
        bn ∈ [true false], 
        pg ∈ [true false],
        up ∈ [:convt :nearest :bilinear]

        if vl == true && up == :bilinear
            #@test_throws
            continue
        end
        model = unet(inchannels = ic, nclasses = nc, volume = vl, base = 4,
            batchnorm = bn, padding = pg, upsample = up, nlevels = 2)
        x, ys = dummy_data(inchannels = ic, nclasses = nc, volume = vl,
            padding = pg, nlevels = 2)
        @test (model(x) |> size) == ys
    end

end