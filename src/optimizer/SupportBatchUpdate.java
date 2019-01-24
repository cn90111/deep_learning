package optimizer;

public interface SupportBatchUpdate
{
	public int getBatchSize();
	public void batchUpdate();
	public void resetBatch();
}
