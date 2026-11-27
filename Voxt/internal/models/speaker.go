package models

// Nemotron 3 Diarization, pinned to the general-access Core ML conversion.
// Only the streaming audio classifier and its learned silence vector are used.
var nemotron3Model = Model{
	ID: SpeakerDiarization, Name: "Nemotron 3 Diarization (fp16)", Kind: Speaker, Runtime: CoreML,
	Family: "nemotron3-diarization", Quant: "fp16",
	Repo: "FluidInference/nemotron-3-diarization-coreml", Revision: "25a90f97f254428d4b30374b76af9c74fdee8327",
	Files: []File{
		{"learnable_sil_emb.bin", 2048, "d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7", true},
		{"monolithic/v2/Nemotron3Diarizer_fast.mlmodelc/analytics/coremldata.bin", 243, "f874278778095b9abfc07106fcbd0e877a8fd5dc34444c7a74331735b41eee68", true},
		{"monolithic/v2/Nemotron3Diarizer_fast.mlmodelc/coremldata.bin", 752, "cc3b262b0ac3028a749943bfe5b2502bd0e5d12c057b0d4d1fd2a2bbf9e4f1b2", true},
		{"monolithic/v2/Nemotron3Diarizer_fast.mlmodelc/model.mil", 501600, "0ead6160e1152656f8c545b113c65d0ee3b73cb88dd96b8050039cdc58904ebf", false},
		{"monolithic/v2/Nemotron3Diarizer_fast.mlmodelc/weights/weight.bin", 198560128, "5efb213341eb1d772d662d852a9a9cf9f45372a599862e16035880742c491967", true},
	},
}
