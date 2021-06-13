function [Cost] = cSum(d, K)

Cost = sum(d);
    if K > 7
        Cost = Cost + K*210000;
    end
    
end