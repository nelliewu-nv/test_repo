
# required environment vairables
#     TIMELOOP_NVIDIA_BASE: pointer to the timeloop-nvidia repo
#     SIM: root directory for storing simulation results

EYERISS_BASE="${TIMELOOP_NVIDIA_BASE}/sparse-optimization/eyeriss-based"
ARCH_FILE="${EYERISS_BASE}/arch/arch.yaml"
MAPPER_FILE="${EYERISS_BASE}/mapper/mapper.yaml"
CONSTRAINT_DIR="${EYERISS_BASE}/constraints"
COMPONENT_DIR="${EYERISS_BASE}/components"
LAYER_DIR="${EYERISS_BASE}/layer_shapes/AlexNet_Sparse"
VAR_DIR="${EYERISS_BASE}/variables"
SPARSE_OPT_DIR="${EYERISS_BASE}/sparse_opt"


OUT_DIR_RELATIVE_PATH="eyeriss"


OUT_DIR_PATH="${SIM}/${OUT_DIR_RELATIVE_PATH}"
if [ ! -d "${OUT_DIR_PATH}" ]; then
    mkdir $OUT_DIR_PATH
fi

for opt_yaml in "${SPARSE_OPT_DIR}"/*
do

  file_name=`basename "${opt_yaml}"`
  file_name_no_ext=${file_name%.*} 
  
  # output directory named using the sparse optimization specification
  if [ ! -d "${OUT_DIR_PATH}/${file_name_no_ext}" ]; then
    mkdir ${OUT_DIR_PATH}/${file_name_no_ext}
  fi
  
  for LAYER_ID in 1 2 3 4 5
  do
    # only run the layer if there is no existing corresponding output folder
    if [ ! -d "${OUT_DIR_PATH}/${file_name_no_ext}/layer${LAYER_ID}" ]; then
      mkdir ${OUT_DIR_PATH}/${file_name_no_ext}/layer${LAYER_ID}
      echo " timeloop-mapper -o ${OUT_DIR_PATH}/${file_name_no_ext}/layer${LAYER_ID} 
                                ${ARCH_FILE} 
                                ${CONSTRAINT_DIR}/* 
                                ${MAPPER_FILE} 
                                ${COMPONENT_DIR}/* 
                                ${LAYER_DIR}/layer${LAYER_ID}.yaml 
                                ${VAR_DIR}/${file_name} 
                                ${opt_yaml}"
                         
      timeloop-mapper -o ${OUT_DIR_PATH}/${file_name_no_ext}/layer${LAYER_ID} \
                         $ARCH_FILE $CONSTRAINT_DIR/* $MAPPER_FILE $COMPONENT_DIR/* \
                         $LAYER_DIR/layer$LAYER_ID.yaml \
                         $VAR_DIR/${file_name} \
                         $opt_yaml
    fi
  done

done

