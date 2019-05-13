using Flux
using CuArrays
# Array -> Image

# Mapping Network
function mapping_network(z::AbstractArray)

    batch_size, z_dim = size(z)

    zb = batch_size * z_dim

    mlp = [Dense(zb, zb) for n in 1:8]

    w = foldl((x, m) -> m(x), mlp, init = vcat(z...))

    return reshape(w, batch_size, z_dim)
end

z = cu(rand(Float32, (1, 512)))
w = mapping_network(z)
print(w)