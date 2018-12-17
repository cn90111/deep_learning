package activation;

public class Relu extends AbstractActivation implements Differentiable
{
	@Override
	public double[] calculate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = weight[i] > 0 ? weight[i] : 0;
		}
		return value;
	}

	@Override
	public double[] toDifferentiate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = weight[i] > 0 ? 1 : 0;
		}
		return value;
	}
}
