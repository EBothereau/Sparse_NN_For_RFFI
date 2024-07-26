
using Flux


#---- Number of TX
Tx = 4 #6, 15 or 16 depending on the database


# ----------------------------------------------------
# --- Sankhe CNN
# ----------------------------------------------------

model = Chain(
    Conv((7,), 2 => 128, pad=SamePad(), relu),                    
    Conv((5,), 128 => 128, pad=SamePad(), relu),                  
    MaxPool((2,)),
              
    Conv((7,), 128 => 128, pad=SamePad(), relu),
    Conv((5,), 128 => 128, pad=SamePad(), relu),
    MaxPool((2,)),

    Conv((7,), 128 => 128, pad=SamePad(), relu),
    Conv((5,), 128 => 128, pad=SamePad(), relu),
    MaxPool((2,)),

    Conv((7,), 128 => 128, pad=SamePad(), relu),
    Conv((5,), 128 => 128, pad=SamePad(), relu),
    MaxPool((2,)),
             
    Flux.flatten,
    Dense(1024, 256, relu), 
    Dropout(0.5),
    Dense(256, 128, relu),
    Dropout(0.5),
    Dense(128,Tx),                                           
    Flux.softmax
)

# ----------------------------------------------------
# --- Hanna CNN
# ----------------------------------------------------

        dr = 0.5 #Dropout rate
model = Chain(
    x -> reshape(x, (size(x)[1], 2, 1, size(x)[3])),
    Conv((3,2), 1 => 8, pad=SamePad(), relu), 
    MaxPool((2,1)),
    Conv((3,2), 8 => 16, pad=SamePad(), relu), 
    MaxPool((2,1)),
    Conv((3,2), 8 => 16, pad=SamePad(), relu), 
    MaxPool((2,1)),
    Conv((3,2), 8 => 16, pad=SamePad(), relu), 
    MaxPool((2,1)),

    Flux.flatten,
    Dense(2048, 100, relu), 
    Dense(100, 80, relu),
    Dropout(dr),
    Dense(80,Tx), 
    Flux.softmax
)

