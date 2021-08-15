@echo off
FOR %%f IN (.\*.cfg) DO (
	python generator.py %%f
)