package activation;

public class Linear extends AbstractActivation implements Differentiable
{
	@Override
	public double[] calculate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = weight[i];
		}
		return value;
	}

	@Override
	public double[] toDifferentiate(double[] weight)
	{
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = 1;
		}
		return value;
	}
}
