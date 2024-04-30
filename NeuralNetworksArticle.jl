using Flux


#---- Number of TX
Tx = 6 


# ----------------------------------------------------
# --- Triple Dense (ours) 
# ----------------------------------------------------

model =Chain(
    Flux.flatten,
    Dense(256*2, 100), 
    Dropout(0.05),
    leakyrelu,
    Dense(100, 64), 
    Dropout(0.05),
    leakyrelu,
    Dense(64,x), 
    Flux.softmax
)

# ----------------------------------------------------
# --- Elmaghbub CNN
# ----------------------------------------------------

model =Chain(
    Conv((4,), 2 => 32, pad=SamePad()), 
    BatchNorm(32),
    leakyrelu,
    MaxPool((2,), stride=2),
         
    Conv((4,), 32 => 48, pad=SamePad()), 
    BatchNorm(48),
    leakyrelu,
    MaxPool((2,), stride=2),
        
    Conv((4,), 48 => 64, pad=SamePad()), 
    BatchNorm(64),
    leakyrelu,
    MaxPool((2,), stride=2),
        
    Conv((4,), 64 => 76, pad=SamePad()), 
    BatchNorm(76),
    leakyrelu,
    MaxPool((2,), stride=2),
        
    Conv((4,), 76 => 96, pad=SamePad()), 
    BatchNorm(96),
    leakyrelu,
    MaxPool((2,), stride=2),
        
    Conv((4,), 96 => 110, pad=SamePad()), 
    BatchNorm(110),
    leakyrelu,
    MaxPool((2,), stride=2),
        
    Flux.flatten,
    Dense(440, 100), 
    Dropout(0.5),
    leakyrelu,
    Dense(100, 64), 
    Dropout(0.5),
    leakyrelu,
    Dense(64,x), 
    Flux.softmax
)

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

# ----------------------------------------------------
# --- Feng CNN-GRU
# ----------------------------------------------------

model =Chain(
    Conv((1,), 2 => 32),  
    BatchNorm(32),
    relu,
    Conv((3,), 32 => 32, pad=SamePad(), relu),  
    BatchNorm(32),
    relu,
    MaxPool((2,)),
    Conv((3,), 32 => 32, pad=SamePad(), relu),  
    BatchNorm(32),
    relu,
    MaxPool((2,)),
    Conv((3,), 32 => 32, pad=SamePad(), relu),  
    BatchNorm(32),
    relu,
    MaxPool((2,)), 
    x -> permutedims(x, (2, 1, 3)),     #layer required to have the correct inputs for the GRU layer           
    Flux.Recur(Flux.GRUCell(32, 50)),
    x-> x[:,end,:],                     #We only take the last output of the GRU layer
    Dense(50, x), 
    Flux.softmax
)
