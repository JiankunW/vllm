#!/bin/bash
curl http://localhost:8000/generate \
    -d '{
    "prompt": "Describe a time when you had to make a difficult decision.",
    "use_beam_search": true,
    "n": 4,
    "temperature": 0
    }'

# # send requests every 1,2,4,... seconds
# counter=1
# while true; do
#     curl http://localhost:8000/generate \
#         -d '{、
#         "prompt": "San Francisco is a",
#         "use_beam_search": true,
#         "n": 4,
#         "temperature": 0
#         }'
#     interval=$((counter ** 2))
#     sleep $interval
#     if [ $counter -eq 5 ]; then
#         break
#     fi
# done