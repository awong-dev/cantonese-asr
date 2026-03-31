# Get download URL to download file
RESPONSE=$(curl -X POST "https://datacollective.mozillafoundation.org/api/datasets/cmn29rqn9016to107eniyak65/download" \
  -H "Authorization: Bearer ${1}" \
  -H "Content-Type: application/json")

# Extract download URL and download file
DOWNLOAD_URL=$(echo $RESPONSE | jq -r '.downloadUrl')
curl -o "Cantonese.tar.gz" "$DOWNLOAD_URL"
