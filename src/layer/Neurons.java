package layer;

import initializer.AbstractInitializer;

public class Neurons
{
	private int linkSize;
	private double[] input;
	private double[] previousLinkWeight;
	private double previousLinkBias;
	private double output;
	private AbstractInitializer initializer;

	public Neurons(AbstractInitializer initializer)
	{
		this.initializer = initializer;
		previousLinkBias = Math.random();
	}

	public Neurons(Neurons other)
	{
		this.linkSize = other.getWeight().length;
		this.previousLinkWeight = other.getWeight();
		this.previousLinkBias = other.getBias();
	}

	public void setLinkSize(int size)
	{
		linkSize = size;
		input = new double[linkSize];
		previousLinkWeight = initializer.initialize(linkSize);
	}

	public void dataIn(double[] data)
	{
		input = data;
		output = calculate(input, previousLinkWeight);
	}

	private double calculate(double[] input, double[] weight)
	{
		double output = 0;
		for (int i = 0; i < input.length; i++)
		{
			output = output + input[i] * weight[i];
		}
		output = output + previousLinkBias;
		return output;
	}

	public double dataOut()
	{
		return output;
	}

	public void updateWeight(double[] weight)
	{
		previousLinkWeight = weight;
	}

	public void updateBias(double bias)
	{
		previousLinkBias = bias;
	}

	public double[] getWeight()
	{
		return previousLinkWeight.clone();
	}

	public double getBias()
	{
		return previousLinkBias;
	}
}
