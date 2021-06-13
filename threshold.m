function [TR_B] = threshold

CC_Sorted=sortrows(CC);
der_1st=imfilter(CC_Sorted,[-1;1],'replicate');
mean_der_1st=mean(der_1st);
der_2nd=imfilter(CC_Sorted,[1;-2;1],'replicate');
TR_B=min(der_2nd(der_2nd>mean_der_1st));

end