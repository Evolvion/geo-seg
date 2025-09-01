setup:
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check src tests

format:
	black src tests

test:
	pytest -q

run:
	python -m geo_seg.train --out_dir runs/debug1

eval:
	python -m geo_seg.eval --ckpt runs/debug1/best.pt

demo:
	streamlit run -m geo_seg.app -- --ckpt runs/debug1/best.pt

clean:
	rm -rf runs __pycache__ src/**/__pycache__ tests/**/__pycache__
