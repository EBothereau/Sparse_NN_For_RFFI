
using Flux


#---- Number of TX
Tx = 6 


# ----------------------------------------------------
# --- Sankhe CNN
# ----------------------------------------------------

model =Chain(
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
    Dense(128,x),                                           
    Flux.softmax
)

# ----------------------------------------------------
# --- Arroyo CNN
# ----------------------------------------------------

model =Chain(
    Conv((10,), 2 => 64, pad=SamePad(), relu),  
    MaxPool((2,)),
    Conv((10,), 64 => 32, pad=SamePad(), relu),  
    MaxPool((2,)),
    Conv((10,), 32 => 16, pad=SamePad(), relu),  
    MaxPool((2,)),
    Flux.flatten,
    Dense(512,64), 
    Dense(64,4), 
    Dense(4,x), 
    Flux.softmax
)

