function [ K ,f, x_values, unf, h,p, KSstatistic, emd] = DistributionEstimate( Data,Norm )
NormDM=Data;
if Norm
NormDM= (Data(:,1) - min(Data(:,1))) / ( max(Data(:,1)) - min(Data(:,1)) );
end
pd = makedist('uniform');
[h,p, KSstatistic, criticalValue] = kstest(NormDM,'cdf',pd);
figure
[f,x_values] = ecdf(NormDM);
unf = unifcdf(x_values)


emd=mean(abs(unifcdf(x_values)-f));
end

