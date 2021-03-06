package layer;

import activation.AbstractActivation;
import initializer.AbstractInitializer;

public class Layer
{
	private Neurons[] neurons;
	private AbstractActivation activation;
	private int linkSize;
	private double[] input;
	private double[] output;

	public Layer(int neuronSize, AbstractInitializer kernelInitializer, AbstractActivation activation)
	{
		neurons = new Neurons[neuronSize];
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i] = new Neurons(kernelInitializer);
		}
		this.activation = activation;
	}

	public Layer(Layer other)
	{
		this.neurons = other.getNeurons();
		this.activation = other.getActivation();
		this.linkSize = other.getLinkSize();
	}

	public void dataIn(double[] data)
	{
		input = data;
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i].dataIn(data);
		}
		output = calculate();
	}

	public double[] getInput()
	{
		return input;
	}

	private double[] calculate()
	{
		double[] output = new double[neurons.length];

		for (int i = 0; i < output.length; i++)
		{
			output[i] = neurons[i].dataOut();
		}
		return output;
	}
	
	public double[] previousActivationOutput()
	{
		return output;
	}
	
	public double[] dataOutput()
	{
		return activation.calculate(output);
	}

	public void setLinkSize(int size)
	{
		linkSize = size;
		for (Neurons neuron : neurons)
		{
			neuron.setLinkSize(linkSize);
		}
	}

	public int getLinkSize()
	{
		return linkSize;
	}

	public int getNeuronSize()
	{
		return neurons.length;
	}

	public void updateWeight(double[][] weight)
	{
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i].updateWeight(weight[i]);
		}
	}

	public void updateBias(double[] bias)
	{
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i].updateBias(bias[i]);
		}
	}

	public double[][] getWeight()
	{
		double[][] weight = new double[neurons.length][];

		for (int i = 0; i < neurons.length; i++)
		{
			weight[i] = neurons[i].getWeight();
		}

		return weight;
	}

	public double[] getBias()
	{
		double[] bias = new double[neurons.length];

		for (int i = 0; i < neurons.length; i++)
		{
			bias[i] = neurons[i].getBias();
		}

		return bias;
	}

	public AbstractActivation getActivation()
	{
		return activation;
	}

	public Neurons[] getNeurons()
	{
		Neurons[] temp = new Neurons[neurons.length];
		for (int i = 0; i < neurons.length; i++)
		{
			temp[i] = new Neurons(neurons[i]);
		}
		return temp;
	}
}
