package initializer;

public class Constant extends AbstractInitializer
{
	private double constant;

	public Constant(double value)
	{
		constant = value;
	}

	@Override
	public double[] initialize(int length)
	{
		for (int i = 0; i < length; i++)
		{
			result[i] = constant;
		}
		return result;
	}
}