package activation;

public class Sigmoid extends AbstractActivation implements Differentiable
{
	@Override
	public double[] calculate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = sigmoid(weight[i]);
		}
		return value;
	}

	private double sigmoid(double weight)
	{
		return 1 / (1 + Math.exp(-1 * weight));
	}

	@Override
	public double[] toDifferentiate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = sigmoid(weight[i]) * (1 - sigmoid(weight[i]));
		}
		return value;
	}
}
