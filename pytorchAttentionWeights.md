# Accessing the intermediate output attention weights in pytorch's transformer modules

For debugging and visualization, it is useful to visualize the attention weight predicted for each input. I couldn't find a great way to get access to this information for pytorch's TransformerEncoder module. My solution, which is kind of messy, but I couldn't figure out something better, involves modifying the `TransformerEncoderLayer` function so that calls to [`MultiHeadAttention` forward](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) functions return attention weights then registering forward hooks to get these intermediate outputs. 

1. Create a module called `myTransformerEncoderLayer` based on [`torch.nn.TranformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html). The main difference in my child class is that I can set the `need_weights` argument to the forward call of the `MultiHeadAttention` block. `myTransformerEncoderLayer` has a variable `need_weights` which is passed to the self-attention block forward call. Here is the code for defining `myTransformerEncoderLayer`:
```
class myTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
  
  def __init__(self,*args,need_weights=False,**kwargs):
    super().__init__(*args,**kwargs)
    self.need_weights = need_weights
  
  def _sa_block(self, x: torch.Tensor,
                attn_mask: typing.Optional[torch.Tensor], key_padding_mask: typing.Optional[torch.Tensor]) -> torch.Tensor:
    x = self.self_attn(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       need_weights=self.need_weights)[0]
    return self.dropout1(x)
  def set_need_weights(self,need_weights):
    self.need_weights = need_weights
```
This is fragile -- if `TransformerEncoderLayer` changes its internal function names, this will not work. 
2. Replace `TransformerEncoderLayer` modules with `myTransformerEncoderLayer`. Here's how my `TransformerModel` class looks:
```
class TransformerModel(torch.nn.Module):

  def __init__(self, d_input: int, d_output: int,
               d_model: int = 2048, nhead: int = 8, d_hid: int = 512,
               nlayers: int = 12, dropout: float = 0.1):
    super().__init__()

    # create self-attention + feedforward network module
    # d_model: number of input features
    # nhead: number of heads in the multiheadattention models
    # dhid: dimension of the feedforward network model
    # dropout: dropout value
    encoder_layers = myTransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)

    # stack of nlayers self-attention + feedforward layers
    # nlayers: number of sub-encoder layers in the encoder
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers,nlayers)
...
```
3. Add a function `set_need_weights` to the `TransformerModel` class that calls each `myTransformerEncoderLayer` layer's `set_need_weights` function:
```
  def set_need_weights(self,need_weights):
    for layer in self.transformer_encoder.layers:
      layer.set_need_weights(need_weights)
```
4. I created the following function for getting the final output of the model as well as the intermediate attention weights. 
```
def get_output_and_attention_weights(model, inputs, src_mask):  

  # set need_weights to True for this function call
  model.set_need_weights(True)

  # where attention weights will be stored, one list element per layer
  activation = [None,]*model.transformer_encoder.num_layers
  def get_activation(layer_num):
    # the hook signature
    def hook(model, inputs, output):
      # attention weights are the second output
      activation[layer_num] = output[1]
    return hook

  # register the hooks
  hooks = [None,]*model.transformer_encoder.num_layers
  for i,layer in enumerate(model.transformer_encoder.layers):
    hooks[i] = layer.self_attn.register_forward_hook(get_activation(i))

  # call the model
  with torch.no_grad():
    output = model(inputs, src_mask)
  
  # remove the hooks    
  for hook in hooks:
    hook.remove()

  # return need_weights to False
  model.set_need_weights(False)    

  return output,activation
```
This makes use of pytorch's forward hooks to get access to intermediate outputs. I found [this tutorial](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/) helpful in figuring out how to do this. 
