build_lore:
	docker build --platform=linux/amd64 -t lore .

run_lore:
	docker run --platform=linux/amd64 -it --rm -v .:/app lore

delete_lore:
	docker rmi lore