function [layer1, layer2] = two_layer_forwardprop(X, W, pool_layer, layer1_act, layer2_act)

layer1 = layer1_act(X * W.'); 

clear W X;    
l2_input = full(double(layer1) * (pool_layer).');    
clear pool_layer;    
layer2 = layer2_act(l2_input);

end
