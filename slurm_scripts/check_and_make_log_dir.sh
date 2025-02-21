LOG_DIR_FILE=log_dir.txt
log_dir=""
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --log_dir)
      log_dir="$2"
      shift # Shift past argument
      shift # Shift past value
      ;;
    *)
      exit 1
      ;;
  esac
done
echo $log_dir > $LOG_DIR_FILE
