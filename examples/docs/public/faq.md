# Frequently Asked Questions

## Is AKL free to run?

Yes. Every component is open source and the reference deployment runs on a single laptop with
Docker Compose. No cloud account is required for the local setup, and the design document
explains how the same architecture scales to petabytes when you do move to the cloud.

## Which document formats are supported?

Markdown, PDF, HTML and GitHub repositories are supported by the built-in connectors. Each
connector implements the same interface, so adding a new source is a matter of writing a
connector class and a parser class and registering both.

## Why is the vector database "derived state"?

Because the Gold layer is the durable record. Vectors are stored in Parquet alongside the text
they were computed from. Qdrant can therefore be rebuilt at any time, and switching embedding
models means writing a new embedding version rather than losing history.
