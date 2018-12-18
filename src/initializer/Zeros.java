package initializer;

public class Zeros extends AbstractInitializer
{
	@Override
	public double[] initialize(int length)
	{
		result = new double[length];
		for (int i = 0; i < length; i++)
		{
			result[i] = 0;
		}
		return result;
	}
}
