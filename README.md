# mhp_lob
Mamba Hawkes Process for Limit Order Book Simulation

Using LOBSTER data (https://lobsterdata.com/home), the MHP can simulate order book events. 
To run the training & simulation, add the orderbook and message csv files to the same folder as the python script. 
Run the following (changing the device and tickers when necessary):

for T in AAPL AMZN GOOG INTC MSFT; do
  python mhp_lob.py \
   --ticker $T --model mhp \
   --baseline bathtub \
   --max-eents 0 --epochs 20 \
   --seq-len 128 --stride 16 --batch-size 64 \
   --time-pred intensity --simulate 5000 --n-sims 200 \
   --device mps --out mhp_lob_sim_${T}
done
