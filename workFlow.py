def RAG_Query(self,query: str, vector_collection_name: str, 
                  RAG_k_local: int = None, 
                  RAG_Fecth_k_local:int  = None ,
                  filtro: models.models.Filter = models.models.Filter()):
        if RAG_k_local == None:
            RAG_k_local = self.conf["RAG_k"]
        if RAG_Fecth_k_local == None:
            RAG_Fecth_k_local=self.conf["RAG_Fecth_k"]

        RQ_qdrant_client = self.DBHook.GetDocsFromDB_ForRetr(vector_collection_name)       
        retrieverQdrant = RQ_qdrant_client.as_retriever(
            search_type="mmr",
            search_kwargs={
                "filter": filtro,
                "k": RAG_k_local,
                "fetch_k": RAG_Fecth_k_local})
        qa = RetrievalQA.from_chain_type( 
            llm=self.Ollama_llm,
            chain_type="stuff",        
            retriever=self.JinaReranker_MiM_retriever(retrieverQdrant),
            chain_type_kwargs={"prompt": PromptTemplate(input_variables=['context', 'question'],template=self.conf["RAG_SysPrompt"])},
            return_source_documents=True
        )
        response = qa.invoke(query)  
        out = {'resultado': None, 'Chunks_Elegidos': None, 'Filtro': filtro}
        # Verificar si el resultado de la invocación no es None antes de acceder a sus claves
        if response is not None:
            out['resultado'] = response['result']
            out['Chunks_Elegidos'] = response['source_documents']
        return out

    def JinaReranker_MiM_retriever(self, _base_Retriever, local_RAG_JinaRR_n:int=None):
        if local_RAG_JinaRR_n == None:
            local_RAG_JinaRR_n = self.conf["RAG_JinaRR_n"]
        adev = _base_Retriever
        if local_RAG_JinaRR_n != 0:
            modelo = HuggingFaceCrossEncoder(model_name=self.conf["RAG_JinaRR_model_path"])
            compressor = CrossEncoderReranker(model=modelo, top_n=local_RAG_JinaRR_n)
            adev = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=_base_Retriever)
        return adev
 def CreaPreguntasDesdeEtiquetado(self, ProyectoOrigen: str, ProyectoDestino: str, bucles: int = 1):
     """ 
     Crea un proyecto LabelStudio espejo a partir de usar un proyecto ya etiquetado usando sus docs como entrada 
     para generara preguntas que sean respondidas por los mismos docs, heredando sus etiquetas orgien.
     El sentido de esta tarea es que dicho proyecto resultado sea usado para entrenar clasificadores de preguntas (queries)
     después de ser supervisadas desde LS.
     Usa el ML que esté configurado en self.Ollama_model, pero se han obtenido mejores resultados con llama3, por lo que 
     es recomendable.
     El prompt es configurable modificando el valor de self.LabSt_CreaPreguntas_SysPrompt
     Bucles es el número de veces que se ejecutará para que genere diferentes preguntas.
     Language : True --> Inglés    False --> Español
     """
     docs_orig = self.LabSt.GetDocsFromLabelStudio(ProyectoOrigen)
    
     docs_questions = list()
     for bb in range(bucles):
         for do in [x for x in docs_orig if len(x.metadata['annotations']) > 0]: 
             new_out = documents.base.Document(
                 page_content=self.Ollama_llm.invoke(
                     self.conf["LabSt_CreaPreguntas_SysPrompt"] + "\n" + do.page_content
                 ),
                 metadata=do.metadata.copy()
             )
             new_out.metadata["original_chunk"] = do.page_content
             docs_questions.append(new_out)

     # Pasar los docs al proyecto destino
     self.LabSt.SendDocsToLabelStudio(sobreescribir=True, NombreProyecto=ProyectoDestino, docs=docs_questions)
    
     # Se seta el label_config destino para que sea igual que el origen
     self.LabSt.GetProyByName(ProyectoDestino).set_params(
         label_config=self.LabSt.GetProyByName(ProyectoOrigen).params['label_config']
     )

    def CrearFiltro(self, query:str, 
                    Clasif_k_local: int = None, 
                    LabelsFromLabelStudio: List[str] = list(), 
                    SelectedLabels: List[str] = list(),
                    BanLabels: List[str] = list(),
                    SelectedSources: List[str] = list()) -> models.models.Filter:
        """ 
        Permite crear un objeto de filtro para el retriever en el esquema de labels que existe en el metadata de los datos importados a la base de datos 
        vectorial desde LabelStudio. Han de pasársele en lista los labels con los que trabaja el proyecto de LS y también permite pasarle manualmente 
        una cantidad de labels fijados por SelectedLabels. Si estos últimos superan la cantidad configurada en Clasif_k_local, la función crea el filtro
        únicamente con SelectedLabels y no hará uso de predicción con el clasificador. Esto es buscado para en el caso de que el usuario quiera fijar 
        ciertos labels para volver a repretir la pregunta.
        """
        mustToAdd = []
        desdeClasificador = list()
        if Clasif_k_local == None:
            Clasif_k_local = self.conf["Clasif_k"]
        if len(SelectedLabels)<Clasif_k_local and len(LabelsFromLabelStudio)>0:
            # si los labels preseleccionados son menos que la configuración de número de labels para la clasificación automática, se le pide al clasificador tantos
            # labels como Clasif_k menos los que ya están en SelectedLabels. Además, también tiene que cumplirse que los labels que vienen de labelstudio tienen
            # al menos uno, ya que esta misma función se podría usar para crear filtros de labels sin necesidad de clasificador.
            desdeClasificador = self.Clasificador(
                query,
                list(set(LabelsFromLabelStudio)-set(SelectedLabels)-set(BanLabels)) # Elimina los labels de LabelStudio si ya están en los preseleccionados así como
                                                                                    # las etiquetas baneadas
                )['labels'][0:Clasif_k_local-len(SelectedLabels)]
        listadoFinal = list(np.append(np.array(SelectedLabels),np.array(desdeClasificador)))
        if listadoFinal:
            mustToAdd.append(models.FieldCondition(key="metadata.annotations",match=models.MatchAny(any=listadoFinal)))
        if SelectedSources:
            mustToAdd.append(models.FieldCondition(key="metadata.source",match=models.MatchAny(any=SelectedSources)))
        return models.Filter(must=mustToAdd)
        