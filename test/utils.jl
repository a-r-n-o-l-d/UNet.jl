using UNet: checksize, minsize, trim, outputsize, adjustsize, padding

@testset "utils" begin
    x1 = rand(32, 32, 4, 6)
    x2 = rand(32, 32, 4, 6)
    @test (chcat(x1, x2) |> size) == (32, 32, 8, 6)

    @test checksize((256, 512), padding = true, nlevels = 4) == (true, true)
    @test checksize((572, 572), padding = false, nlevels = 4) == (true, true)
    @test checksize((256, 500), padding = true, nlevels = 4) == (true, false)
    @test checksize((572, 512), padding = false, nlevels = 4) == (true, false)

    @test minsize(true, 4) == 64
    @test minsize(false, 4) == 188

    @test trim(1, 4) == 88

    @test outputsize((512, 512), true, 4) == (512, 512)
    @test outputsize((572, 572), false, 4) == (388, 388)

    @test adjustsize((256, 256), true, 4) == (256, 256)
    @test adjustsize((388, 388), false, 4) == (572, 572)
    @test adjustsize(4, true, 4) == 64
    @test adjustsize(4, false, 4) == 188

    res = (((92, 92), (92, 92)), ((0, 0), (0, 0)))
    @test padding((388, 388), padding = false, nlevels = 4) == res
    res = (((0, 0), (0, 0)), ((0, 0), (0, 0)))
    @test padding((256, 256), padding = true, nlevels = 4) == res
end