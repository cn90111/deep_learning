package dataset;

import java.io.FileNotFoundException;

public abstract class Dataset
{
	protected double[][] trainFeature;
	protected double[][] trainLabel;
	protected double[][] testFeature;
	protected double[][] testLabel;
	
	public Dataset() throws FileNotFoundException
	{
		arrayInit();
		loadData();
	}
	
	protected abstract void arrayInit();
	protected abstract void loadData() throws FileNotFoundException;
	
	public abstract double[][] getTrainFeature();
	public abstract double[][] getTrainLabel();
	public abstract double[][] getTestFeature();
	public abstract double[][] getTestLabel();
}
