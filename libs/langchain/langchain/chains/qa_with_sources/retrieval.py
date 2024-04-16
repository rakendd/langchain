"""Question-answering with sources over an index."""

from typing import Any, Dict, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
import os
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
import time

class RetrievalQAWithSourcesChain(BaseQAWithSourcesChain):
    """Question-answering with sources over an index."""

    retriever: BaseRetriever = Field(exclude=True)
    """Index to connect to."""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_documents_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    # def _get_docs(
    #     self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun
    # ) -> List[Document]:
    #     question = inputs[self.question_key]
    #     docs = self.retriever.get_relevant_documents(
    #         question, callbacks=run_manager.get_child()
    #     )
    #     docs =  self._reduce_tokens_below_limit(docs)
    #     print("in _get_docs")
    #     print(docs)
    #     return docs
    #     # return self._reduce_tokens_below_limit(docs)


    # def _get_docs(self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:

    #     question = inputs[self.question_key]
    #     before_time = time.time()
    #     docs = self.retriever.get_relevant_documents(question, callbacks=run_manager.get_child())
    #     print("Retrieved docs time: ", time.time() - before_time)
    #     # cohere_api_key = 
    #     # before_time = time.time()
    #     # co = cohere.Client(os.environ['COHERE_API_KEY'])
    #     # print("before cohere")
    #     print(docs)
    #     # valid_docs = [doc.page_content for doc in docs]
    #     # reranked_docs = co.rerank(model="rerank-english-v3.0", query=question, documents=valid_docs, top_n=5, return_documents=True)
    #     # print("Cohere time: ", time.time() - before_time)

    #     # reranked_docs_indices = [i.index for i in reranked_docs.results]
    #     # print(reranked_docs_indices)
    #     # docs_c = [docs[i] for i in reranked_docs_indices]
    #     # print("after cohere")
    #     # print(docs_c)

    #     # bge reranker
    #     doc_pairs = [[question, doc.page_content] for doc in docs]
    #     before_time = time.time()
    #     scores = reranker.compute_score(doc_pairs, normalize=True)
    #     print("bge time: ", time.time() - before_time)
    #     print("scores", scores)


    #     # Combining scores with documents for sorting
    #     scored_docs_with_indices = list(zip(scores, docs, range(len(docs))))

    #     scored_docs_with_indices.sort(reverse=True, key=lambda x: x[0])

    #     # Extracting sorted documents and indices
    #     sorted_docs = [doc for score, doc, idx in scored_docs_with_indices]
    #     sorted_indices = [idx for score, doc, idx in scored_docs_with_indices]

    #     print(sorted_docs)
    #     print(sorted_indices)

    #     # Optionally reduce the number of documents if needed
    #     docs = self._reduce_tokens_below_limit(docs)
        
    #     final_context_parts = []
    #     for doc in docs:
    #         # Basic document summary
    #         doc_summary = [
    #             f"- Model Name: {doc.metadata.get('nodeName', 'Unknown')}",
    #             f"  Type: {doc.metadata.get('type', 'Unknown')}",
    #             f"  Description: {doc.page_content.strip()}",  # Assuming `page_content` ends with a newline
    #         ]
            
    #         # Process dependencies, if any
    #         if 'dependenciesDetails' in doc.metadata:
    #             dependency_summaries = []
    #             for dependency in doc.metadata['dependenciesDetails']:
    #                 dependency_name = dependency.get('name', 'Unknown Dependency')
    #                 dependency_type = dependency.get('type', 'Unknown Type')
    #                 dependency_desc = dependency.get('description', 'No description available.')
                    
    #                 # Format each dependency summary
    #                 dependency_summary = f"  - {dependency_name} (Type: {dependency_type}): {dependency_desc}"
    #                 dependency_summaries.append(dependency_summary)
                
    #             if dependency_summaries:
    #                 doc_summary.append("  Depends On:")
    #                 doc_summary.extend(dependency_summaries)  # Add all dependency summaries
            
    #         final_context_parts.append("\n".join(doc_summary))
        
    #     # Combine the individual summaries into a final context
    #     final_context = "\n\n".join(final_context_parts)
    #     print("Final Context Sent to GPT:", final_context)
        
    #     return final_context
    
    def _get_docs(self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        question = inputs[self.question_key]
        before_time = time.time()
        docs = self.retriever.get_relevant_documents(question, callbacks=run_manager.get_child())
        # print("Retrieved docs time: ", time.time() - before_time)
        # print(docs)

        # Assuming reranker is defined elsewhere in your class
        doc_pairs = [[question, doc.page_content] for doc in docs]
        before_time = time.time()
        scores = reranker.compute_score(doc_pairs, normalize=True)
        # print("bge time: ", time.time() - before_time)
        # print("scores", scores)

        # Combining scores with documents for sorting
        scored_docs_with_indices = list(zip(scores, docs, range(len(docs))))
        scored_docs_with_indices.sort(reverse=True, key=lambda x: x[0])

        # Extracting sorted documents
        sorted_docs = [doc for _, doc, _ in scored_docs_with_indices]

        # Optionally reduce the number of documents if needed
        # Ensure this method returns a list of Document objects
        reduced_docs = self._reduce_tokens_below_limit(sorted_docs)

        # Assuming _reduce_tokens_below_limit returns a list of Document objects
        # If you need to update the content or metadata of the documents based on the processing done above,
        # you would create new Document objects here. For example:
        updated_docs = []
        for doc in reduced_docs:
            # Update document content or metadata as needed
            # This is a placeholder; actual updates depend on your requirements
            new_metadata = doc.metadata.copy()  # Assuming you want to update metadata
            new_metadata['processed'] = True  # An example update
            updated_doc = Document(page_content=doc.page_content, metadata=new_metadata)
            updated_docs.append(updated_doc)

        # print("updated docs")
        # print(updated_docs)
        return updated_docs

    async def _aget_docs(
        self, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun
    ) -> List[Document]:
        question = inputs[self.question_key]
        docs = await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        return self._reduce_tokens_below_limit(docs)

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retrieval_qa_with_sources_chain"
