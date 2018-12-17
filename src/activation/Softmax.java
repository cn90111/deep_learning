package activation;

public class Softmax extends AbstractActivation implements Differentiable
{
	@Override
	public double[] calculate(double[] weight)
	{
		double sum = 0;
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			sum = sum + weight[i];
		}
		for (int i = 0; i < weight.length; i++)
		{
			value[i] = weight[i] / sum;
		}
		return value;
	}

	@Override
	public double[] toDifferentiate(double[] weight)
	{
		double sum = 0;
		double[] value = new double[weight.length];
		for (int i = 0; i < weight.length; i++)
		{
			sum = sum + weight[i];
		}
		sum = Math.pow(sum, 2);

		for (int i = 0; i < value.length; i++)
		{
			for (int j = 0; j < weight.length; j++)
			{
				if (i != j)
				{
					value[i] = value[i] + weight[j];
				}
			}
			value[i] = value[i] * weight[i];
			value[i] = value[i] / sum;
		}
		return value;
	}
}
