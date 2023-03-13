from transformers import FillMaskPipeline, TokenClassificationPipeline


class OVPipeline:
    def __init__(self, *args, **kwargs):
        self.padding = kwargs.pop("padding", "do_not_pad")
        self.max_length = kwargs.pop("max_length", None)
        self.truncation = kwargs.pop("truncation", None)
        super().__init__(*args, **kwargs)


class OVFillMaskPipeline(OVPipeline, FillMaskPipeline):
    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(
            inputs,
            return_tensors=return_tensors,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
        )
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs


class OVTokenClassificationPipeline(OVPipeline, TokenClassificationPipeline):
    def preprocess(self, sentence, offset_mapping=None):
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=self.truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            padding=self.padding,
            max_length=self.max_length,
        )
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        return model_inputs
