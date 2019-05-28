using Flux
using CuArrays
# Array -> Image

# Mapping Network
function mapping_network(z::AbstractArray)
    # We need to know the shape of our input to construct the network layers
    batch_size, z_dim = size(z)

    # Multiply the batch size by the latent dimension to to get the size
    # of the matrix representation
    zb = batch_size * z_dim

    # Build a list of 8 Dense/Fully Connected layers to make the Mapping Network from the paper
    mlp = [Dense(zb, zb) for n in 1:8]

    # Fold over the list of layers, passing in the matrix representation of the Z vector
    w = foldl((x, m) -> m(x), mlp, init = vcat(z...))

    # Return the output of the Mapping Network, reshaped back into the original
    return reshape(w, batch_size, z_dim)
end

function main()
    z = cu(rand(Float32, (1, 512)))
    w = mapping_network(z)
    print(w)
end
