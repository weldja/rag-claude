#!/bin/bash
mkdir -p certs
openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout certs/server.key \
    -out certs/server.crt \
    -subj "/C=GB/ST=England/L=London/O=Weld AI/OU=RAG/CN=ragassistant" \
    -addext "subjectAltName=IP:127.0.0.1,IP:0.0.0.0,DNS:localhost,DNS:ragassistant"
echo "Certificate generated in certs/ — valid for 10 years"
