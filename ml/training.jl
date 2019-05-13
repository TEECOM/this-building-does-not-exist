using Flux
using CuArrays
# Array -> Image

# Mapping Network
function mapping_network(z)
    mlp = [Dense(512, 512) for n in 1:8]

    return foldl((z, m) -> m(z), mlp, init = z)
end

z = rand(512)
w = mapping_network(z)
print(w)