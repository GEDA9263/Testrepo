??
? ?
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_9*
value_dtype0	
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name60146*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?'*
dtype0*??
value??B???'BaB<start>B<end>BonBofBtheBinBwithBandBisBmanBtoBanBsittingBtwoBstandingBatBpeopleBareBnextBwhiteBwomanBholdingBthatBsomeBpersonBlargeBdownBtopBgroupBstreetBtableBsmallBtennisBupBridingBhisBfrontBnearBblackByoungBtrainBdogBredBbyBplayingBwhileBhasBbaseballBplateBcatBwalkingBroomBblueBit.BparkedBgreenBbathroomBthereBkitchenBboyBthreeBpizzaBlookingBcoupleBforBbusBsignBfieldBmenBitBfoodBsideBfromBotherBclockBflyingBoutBfield.Btable.BwaterBseveralBlayingBstreet.BplayerBoverBgrassBbedBoneBgirlBwoodenBtheirByellowBeatingBcityBsnowBwearingBsitsBthroughBherBasBbrownB
skateboardBaroundBtoiletBpictureBballBhorseBbeachBlivingBcoveredBthisBgiraffeBmanyBeachBcloseBunderBgameBfilledBoldBwater.BbuildingBlittleBtruckBveryBbearBfireBoutsideBopenBcomputerBintoBcakeBelephantBchildBboardBairBlaptopBroadBbehindBbenchBitsBfrisbeeB	differentB	building.BstopBbunchBareaBstandsB
motorcycleBviewBphoneBdeskBbirdBbigBbackBcarBanotherBtallBumbrellaBwomenBsinkBtrafficBinsideBlightBcellBroad.BgiraffesBstandBbatBalongBteddyBwindowB	surfboardBkiteBphotoBairplaneBroom.Bplate.BparkBvaseBglassBtogetherBsheepBhydrantBskisBdirtBsandwichBplaneBboatBreadyBbeingBhorsesBbowlBball.BfullBtreeBorangeBtowerBstuffedBcounterBcarsBparkingBbeach.BgrassyBshirtBoffBarea.BimageBskateBhandBcowsBtakingBdrivingBbabyBacrossBaboveBzebraBwallBsitBpinkBflowersBsnowyBskiBsnow.Bgrass.BdoingB	elephantsBbackground.BhotBcouchBholdsBcourtBzebrasBbesideBtreesBhangingBskiingBlongBtrees.BmiddleBfourBemptyBbananasBgoingBwiiBracketBduringBcourt.BwaitingBchairBwaveBpieceBsignsBvideoBsoccerBpairBgrazingBcuttingBwineBdoubleBbikeBheadBgettingBlotBherdBwhoB
vegetablesBtoppedBsuitBskateboard.BtracksBfenceBwindow.Bsky.BglassesBvariousBtree.Bgame.BcowBsmilingBfloorBday.B	travelingBstoveBjumpingBwatchingBcake.Bocean.BdisplayBanimalsBsidewalkBlotsBfood.BbeBwall.BlooksBfrisbee.BswingingBhillBbirdsBgroundBbed.BusingBhatBchildrenBwoodBplayersBbearsBsurfBoceanBkitchen.BcarryingBrefrigeratorBcolorfulBpark.Bit'sBmirrorBskierBagainstBtieBskyBhitBtalkingBmetalB	umbrella.BposingBbodyBthem.BsurferBphone.BguyBfruitBfence.BtrickBkitesBaboutBcornerBsetBmaleBchairsBslope.BladyBjetBallBbroccoliBluggageBkeyboardBheBcutB	passengerBmadeBhomeBother.BnumberBbrickBlineBtrackBrunningBridesBmountainB
backgroundBairportBstationBsomeoneBremoteBhalfBdressedBdogsBcrowdBbusyBdeckerBstation.BpullingBpoleBpaperBbananaB
surroundedBcheeseBcatsBskiersBskateboarderBshowingBlyingBlikeBanimalBtracks.B	together.B
surfboard.BsliceB	preparingBdesk.Bcounter.BwalkBboxBovenBhill.BboatsBwalksBsign.BplatesBmotorcyclesB	chocolateBbicycleBwave.BthrowingBboysB	sidewalk.Bboard.Bair.B	umbrellasB	somethingBpizza.BhighB	decoratedBdarkB	displayedBadultBtop.B
televisionBmirror.BlookBbedroomBmakingBgrayBgirlsBfaceBswingB	snowboardBmeatBsurfingBsink.BdoorBtvBsleepingBkidsBhaveBfloor.BfewBcupBcloudyBtruck.BtowardsBlightsBlaptop.Bkite.BsnowboarderB
restaurantBpileBcity.Bcar.BshowerBplayBoutdoorBhavingBground.BdonutsBcoloredBbench.BslopeBpurpleBjacketBitemsB	bathroom.BattachedBwayBthemBside.BnightBcrossB
buildings.B	buildingsBrowBlushBkidBhouseBhittingBtrayBon.BshownBbat.BappleBbookBbetweenBbagBsilverBplasticBolderBlot.BlinedBbatterBramp.BfruitsBstatueBpastBontoBgreyBcrossingB	computer.BcoffeeBcameraB	beautifulBvasesBscissorsBriverBpicturesBoutside.BofficeBhand.BdrinkingBdog.BcomingBwetBsceneBhandsBflowerBnoBmouth.BleaningBintersectionBstoppedBseenBperchedBcarrotsBbusesBbottleBtoyBsunnyBscreenBsandyBrestingBpaintedBorBbirthdayB
underneathBracket.BpublicBfreshBfemaleBairport.Btoilet.BmouseBmotorcycle.BmealBstoneBplanesBmovingBmotorB	microwaveBshowsBlight.BcatchBtrack.B	mountain.BcountryBcleanBbrightBriver.BpiecesBpassingBorangesBlitBjustBchair.BbrushingBtrain.Btoilet,Bsink,BsingleBsandBplacedB
photographBpeople.BmodernBdiningBdayBcanBwavesBvegetables.BrailBendBcamera.BbreadBbowlsBbeenBsuitcaseBoven.Bhorse.BbridgeBbathBtheyBhim.BguysBbus.BnintendoBfeedingBfamilyBedgeBweddingBuniformBteethBstoreBskateboardingBnotB	furnitureBcattleBtypesBskis.BrunwayBleavesBlapBdressBdonutBcaseBwellBfriesBdishBcabinetsBbenchesBbelowBwatchBtrucksBtablesBsaladBracquetBhouse.BgatheredB
containingBcartBworkingBtubBtimeBslicedBsauceBpreparesBpitcherBmountedBforest.BfiveBfencedBbowl.BtakeBslicesBrunway.Bracquet.BpanBothersB	mountainsB	keyboard.BcookingBcatcherBbear.B	baseball.B	airplanesBtennis.BprettyBpostBniceB	includingBhelmetBhairBcouch.B	computersBbananas.BzooBtiledBsheBmarketBflowers.B
enclosure.B	elephant.BboxesBbaseBtowardBreachingBpointingBnight.BmonitorBcuteB.Btie.BsteelBshotBrideBrestaurant.BrampBmouthBeatBdoor.BdinnerBcontrolBclocksBcarriageBboardsB
umbrellas.BtryingBtower.BsunB
snowboard.BshelfBpicture.BfloatingBengineBbikesBworkBrefrigerator.BrainBplantBpathB
passengersBlakeBhydrant.BfrenchB	containerBclearBapplesBvintageBvarietyBtricksBtileB
surfboardsBsinksBnearby.BflatBcurbBblanketBbasketBatopBwhereBtowelBtarmacBsettingBservingBseatBpoliceB
performingBmultipleBloneBlake.Bhands.BflyB	enclosureBdrinkBdoughnutB	clutteredBwireBwideBstripedBrocksBrockBpizzasBknifeBhimBdrawnB
commercialBafterBtv.BtoddlerB	stainlessBrailroadBjumpBgetBfliesBenjoyingBdryBbeerBasianBtakesBpulledBposesBkites.Bitems.BhotelBhat.BgraffitiBgearBfeetBclock.B	cellphoneB
assortmentBwine.BswimmingBstove.BsnowboardingBsameBplantsBmeat,B	equipmentBchurchBawayBarmBwindowsBup.Btraffic.BrightB
reflectionBplatformBleftBforkB	doughnutsBbrushBbooksBboarderBadultsBwhichBtelevision.BstickingBplaysBpaintingBmilitaryBlegsBiceBdrivesBbothBwallsBtrainsBtoppingsBswingsBskateboardsBshopBroundBprofessionalBpole.BmeterBlowBfishBdirtyB
decorativeBcorner.ByardBwalls.Bvase.B	suitcase.BshortsBpotBpitch.BmidBhugeBgamesBfriendsBcementBcandlesBbeforeB
appliancesB	vegetableBtownBstaringBsoupBservedBschoolB
sandwichesBsand.BposeBhotdogBgoBgiantB	cabinets.B	broccoli.BbottomBbakedBwatchesBvehicleBtrunkBtrailerBtourBstyleBstop.BstillBstackedBspaceBsmilesBreadsBpolarBplaceBlaysBlawnBkindBglasses.BflockBfancyBcarrots,BcabinetBbackpackBappearsBrockyBresidentialBrackBpicnicBphoto.B	fireplaceBfacingBdishesBdessertBcrowdedBconcreteBbrokenB	broccoli,BbathtubBback.Bwii.BwasBtrick.BtomatoesBteamBtanB	structureBshelf.BserveBseatedBrain.BmotherBmessyBeatsBeatenBdoesBcreamBchickenBcars.BbrideBbox.Bboat.BboardingBbedsBwoodedBwildBtub.BtoiletsBthrowBsurface.Bstore.BslidingBshapedBseeBriderBreadingBmakeBjumpsBfruit.BfridgeB
electronicBdrinksBchairs.BbottlesBbananas,BtransitBtoothBtarmac.BtakenBsofaBsigns.BseaBscooterBriceBrestroomBputtingB	propellerBpoolB	platform.BphotosBpersonsBpastureBnewBneckBnarrowBlampBgivingBdirt.Bbridge.BbiteBbike.BbeneathBbagsBassortedBzoo.BvehiclesBturnBtrailBskatingBsixBpartBpansBopen.B
mountains.Bmatch.Bman.BmakesBluggage.BlaptopsBladiesBknife.BkindsBfootBfoodsBface.Bdisplay.B
controllerBconstructionB
collectionB
cellphone.BcatchingBcarrotBbusinessBbunBbook.BblurryBanimals.BactionBwoman.BusedBsurfersBsurfaceBsubwayBstairsBskiesBshowBshore.BshoesBsaysBremote.BrainyBpreparedBmarbleBlunchBliesBleansBglass.Bfries.B	fashionedBdown.BcenterBbunchesBbicyclesBarrangedBantiqueB	airplane.BwhatBtowelsBthingsBstove,B
snowboardsBshortBshelvesBruralBripeBracingBpillowsBmonitorsBloadedBhighwayBhardBgrowingBgroomBgrillBgiraffe.BelectricB	distance.BcontrollersBcontroller.BcloseupBcheese,BcargoBby.BbranchBblanket.Bbicycle.BaloneBwatch.BuniformsBtraysBthrowsBtheseBtable,B	suitcasesB	scissors.BrunBonlyBonionsBmeter.BmatchBmachineBleadingBkittenBintersection.BframeBfallingBdoorwayBdeviceBcoatBclothesBcat.BbushesBarmsBwrittenBwrappedBwaitsBumpireBtrashB	toppings.B
sunglassesBsquareBsidesBshower.BrosesBrollingBrollBpullsBpolesBpitchBpickBpastriesBoutfitBmodelBloadingBkneelingBjetlinerBheadsBhead.BdistanceB
displayingBcow.BcouchesBcookedBcontrollers.BclosedBcheese.B	breakfastBbooks.BbarBaround.BamongByard.Bwaves.BvanBusesBurbanBunmadeB
toothbrushBteeth.BsteepBstallB	someone'sBsmileBrugBrocks.BraceBpotsBplatterBpeppersBpedestriansBoverlookingBleaves.BislandBholdB	hillside.Bher.BhangsBgrazeBfarmBdeadBcrowd.BceilingBbeigeBbed,BbasketsBbag.Bwoods.BwoodsBwinterBwindows.BweBwatches.BurinalsBtopsBtomatoBsmartBsizedBshoreBshadeBsetting.Bseat.B	sandwich.BputBpieBphonesB	partiallyBpaddleBornateBlaidBincludesBhowBhousesBheldBgoldBfront.BforestBfloorsBfeaturesBdoorsBdockBdesertBvestBsitingBsemiBsauce.BrestsBrestBred,Bpoles.Bpath.Bpaper.BpantsBoverheadBmotorcycles.Bmarket.BlandBketchupBjuiceBjetsBgardenBfrostingBfighterB	featuringBdockedBdesktopBcostumeB
containersBclose-upBbowBapproachingBwritingB	watching.BveggiesBtypeBtowBtouchingBtime.BtalksBsticksBsteamB	sculptureBrock.BridersBrice,Brail.BpostedBphones.BpenB
pedestrianBpavedBpassengers.BpairsBneatlyBmonkeyBmetersBleatherBlandingBiBhome.BgreatBgoesBgetsBgateBfunBfoilBflagBfastBfakeBeitherBdonuts.BdisplaysBcouch,Bbedroom.Bappliances.BwithoutBtoysBtown.BtiesBstuffBstageBsnowboardersBsmallerBshoppingBshinyBplane.BpettingBoff.B	mushroomsBmetallicBline.BledgeB	keyboard,Bhorses.BholeBfloralBflipBfarBevent.BcommuterBbuiltBbrush.BbranchesBbranch.BblowingBbaconButensilsBtray.BtravelsBtinyBteaBsweaterBsuitsBsteakBstand.BsodaBsoccer.Bskiing.Bscreen.B	sandwich,BsailingBracketsB
practicingB	potatoes,BpotatoesBplace.BpetBperson.Bperson'sB	pepperoniBpasture.BpastaBpartyBout.BnoseBmopedBmonitor.BkickingBinteriorBinside.BifBhimselfBhayBhatsBfrisbeesBfriedB	formationBfork.BeyesBenclosedBear.BcurtainBcurb.BcupsB	computer,BcollageB	christmasBcelebratingBbutBbanana.BaircraftBwhite.BwetsuitBwalkwayBwagonBvegetables,BvanityBturnedBtravelBtractorBtilesBsurroundingBsubBstreetsBspreadBsoBsmokingBshop.BsheetsBreturnBrelaxingBrailingBproppedBpowerBplants.BperformsBpassBpapersBpackedBofficersBobjectBmouse.B
microwave.Blights.Blap.BjeansBitemBhitsBhelpingBglovesBglazedBgate.B
furniture.BforeignB
fireplace.BeventBelderlyBeggsBeating.BdesignBdeepBcontrol.BcolorsBchairs,Bcase.BcandleBbutterBbushBbrightlyBblockBblenderBbeverageBbenBbattingBbathtub.BbarnB	apartmentBanimal.ByouBwiresBsurfboards.Bsunset.BstuntBstickB
spectatorsBspace.BsmokeBskirtBskateboardersBshoeBshipBshallowBseparateBsellingBsaleBroof.BroofBrefrigerator,BrearBrack.BpuppyBpondBpointsBpitchingBpersonalBparadeBpan.B	organizedBoranges.BoppositeBopenedBoffice.BnumerousBnetBmoundBmobileB	miniatureBminiBlettuceBlegBleashBlayerBlawn.Blarge,BjarsBhoseB	horsebackBhomemadeBhardwoodBhangBgloveB	gentlemanB	gatheringBfries,BflightBfirstBfansB
equipment.B
elephants.Bdogs.BdoBdiscBdeltaBcupcakesBcows.Bcones.BclothingBcheckingBcalfBcakesBbrushesBbaldBbaggageBapples.BairlinerBadvertisingBwhite,BwateringBwalkway.BvisibleBtoweringBterminalBtailBtabbyB
something.BsoldiersBsofa.BsizeBsignalBsheep.BsharingBseagullBsalad.BsailBrailwayBproduceBpottedBpool.Bpond.BpineBpickupBpeople,BpassesB	outdoors.B	observingBmugBmirror,Bmeal.BmagnetsB
locomotiveBlinesBlidBlickingBlampsBladderBjerseyBindoorBimagesBhuggingBhay.BhappyBhair.BgoldenBgasBgarbageB	followingBfluffyBflagsBfishingBentranceBelectronicsBdriveBdrink.BdeliveryBcup.B
countertopB	connectedBcoffee.Bchild'sBchair,BcardBcapBcalmBburgerBbreakBbouquetBbeer.Bbarn.BbareBbar.BalsoBwoman'sBwheelBvehicle.BupsideBtub,B
travellingBsuppliesBsunsetB
structure.BstickersBsteepleBspoon.BspoonBsniffingBslopesBsillBsign,Bshown.BshakingB	scatteredBroseBrackets.BpushingBpillows.BpierBpavementBpastryBparade.BpalmBpackBopeningBneonBmustardBmoveBmoreBmissingBmB
lighthouseBlightedBlamp.B
industrialBhereBheadedBhallwayBgrilledBgravelBgoodBfrostedBfootballBearBdonut.Bdog,Bdish.B	directionBdecorationsBdeckBcutsBcupcakeBconveyorB
computers.B	comforterBcolorBclothBclayB	cigaretteBchefBcauliflowerBcarriesBcandles.B	cabinets,Bbushes.Bbottle.BbendingBbenches.Bbatter,BbacksB
attemptingBarrangementBalleyBaerialBworkersBway.ButilityBtrail.Btoothbrush.B	tomatoes,BtinBthanB	telephoneBteamsBtastingBsuvBsurfing.Bsun.BstuckBstrangeBstepsBsonBsomeonesBsmiling.Bski.BskaterBshapeBselfieBsectionBsayingBsausageBrunsBrug.BreachesBquietBpurseBposedBplates.Bplant.BpillowBpiledB	pictures.BpatioBpartlyBoutdoorsBofficerBobjectsBnicelyBnameBmultiBmuddyBmostlyBmatchingBmarkedBmachine.BlocatedBlightingBleagueBlayBlaptops.BkickBjump.BjumboB
individualBhotdogsBharborBgoatsBgoatBfreezerB
formation.BformalBfloodedBeyeBdoublesBdoorway.B
displayed.Bdesk,BdaughterBcrustBcowboyBcoveringBcoverBcourse.B
consistingBcompetition.BcloudsBcloth.BclassicBcenter.B	cardboardBbun.BbucketBbitingBbeefBbears.BbathingBbankBartBamericanBairlinesBabove.B3B2BwindBwholeBwheelsBwereBwater,BwarningBviewedBuseBtrees,BtossingBthickBtentBtaxiBtapeBtankBt-shirtB	suspendedBsurf.BstripBsteps.BstaresBsportsBshoes.BshirtsBshiningBsheetBseriesBseemsBrustedBraisedBpurse.BpugBprogressBposterB
positionedBplaidBpizza,BpicturedBphone,Bpen.B	pavement.Boranges,BnaturalBmulticoloredBmonitor,BmixedBmeltedBmatBmalesBmacaroniBlogsBloadBleadsBjuice.BjarBit,BindividualsBin.BheavyBheadingB	hamburgerBgolfBgatherBgarageBfridge.BframedBforeground.BfoldedBfocusB
electricalBdrawingBdowntownBdouble-deckerB
directionsBdimlyBdesignedB	depictingBcubicleB	crosswalkBcratesBcoversBcontainsB
container.BcircularBchurch.B	checkeredBcartsBcarrots.B	carriage.BcandyBbussesBbread.BbellBbehind.B
basketballBattireBasleepBarea,BaluminumB	alongsideB"Bwires.BwingsBweirdBwalk.BusBunusualBturningBtrunksBtoppingBtongueBthinBthat'sBtalkBtagsBtables.Bswing.Bsuit.Bstreets.Bstairs.BspaciousBsnackBslicingBskies.B	shirtlessBsetsBrowsBrice.BreallyBrangeBpost.BpipeBpillow.BpeelingBpeelBpedestalBpatchBparrotBovercastBoven,Bobjects.BnoteB	newspaperBnearbyBmushroomBmovesBmotionBmarriedBman,Bledge.BleapingBleafyBlaneBlambBkittyBjockeyBingredientsBincludeBhole.Bhighway.BhealthyBhairedBhaB	graffiti.Bgear.BfurryB	furnishedBfunnyBfreshlyB
foregroundBflownBfloors.BfixingBfield,BfeedsBfallBfacesBdrinks.B	doughnut.BdippingBdinner.BdessertsBdesksBcycleBcups.Bcountryside.BcountersBcouches,BchiliB	children.Bchild.BchasingBcatchesBcarpetBcake,BbuyingBbunnyBbooks,BblondeBbentBbeltBbeardBbeach,Bbasket.Bbase.B&BwitBwingBwearsBwaxBwaitBwadingBuse.B	uniformedBtypingBtriesBtrailer.Btoy.B	tomatoes.BtoastedBtentsBtellingBteachingBtaggedBstringB
stretchingBstreet,BstoryBstomachBstairBsortBsleepsBsizesBsimpleBsigns,Bsides.Bshow.Bshirt.Bserved.BsausagesBsalad,BrollsBrolledBremotes.BrelishBreachBrainbowBrabbitBproduce.BprintBpositionBporchBplate,Bpipe.BpigeonsBpatternBparty.BpaddlingB	overhead.B	operatingBone.BoilB
microwave,B
microphoneBmen.BmeadowBmansBlifeBlieBleavingBlargerBlaptop,Blamps.Bkitchen,BkettleBkeptBinterestingBindianBhouses.BhoagieBhills.BhillsBhelpsBheartBheardBhandleBguardBgrindingBgrapesBgivesBgirl.Bgarden.Bgames.BfrozenBfloor,BfigureBfatBfanBentertainmentBenterBeggBdownhillBdock.BdirectlyBdipBdesignsBdaytime.Bdark.Bcurtain.BcurledBcrownBcookiesBconesBclusterBclimbingBcircleBchipsBchewingBchest.BceramicBcaughtBcastleBcakes.Bcabinet.BbullBboy.Bbowls.BboothB	bookshelfBboards.BboardersBblue,BbeansB	backpacksB	backpack.Bapples,BamountB	abandonedBworkerBwomen.BweatherBwearBviewingBview.B	utensils.BurinalBuponBup,BtwinBturn.BturkeyBtrunk.BtrolleyBtransportationBtouchesBtieredBthumbsBtelevision,BteenagerBteeBsystemBswungB
suitcases.B	stretchesBstretchBstream.BstrawBstraightBstorageBstoppingB	stoplightBstickerBsteppingB
stationaryBstareBstableBst.BsprayingBspinachBsoldierBsmellingBslopes.BslideBsleepBskiisBskiierBsinks.BsignageBsidewaysBshuttleBshinesBsewingBsecondBseatsBscarfBrusticBropeBroamingBroadway.BroadwayBright.BriddingBremotesBrefereeBreadBrange.BrandomBrailing.BracerBprivateBprepareBpracticeBpouringBpotatoBponyBplentyBpickingBpepperBpeekingBpastry.BpartsBparaBpancakesBpaintBownerBowlB	onlookersBonionBobstacleBmuseumBmulti-coloredBmuchBmotorcyclistsBmitt.Bmid-airBmessageB	measuringBmattressBmaskBmashedBliquidBlibrary.BitselfBironBinteractiveBindoorsBherselfB
headphonesBhalf.BhadBgroupsBgrass,BgraniteBgrandfatherBglove.BgamingBfruit,B	frosting.BfreightBfreeBforksBfollowedBflower.BfingerBfightingBfatherBfarm.BextraBeveningBenjoyBempty.BelegantBeaten.BeagleBducksBduckBdriverBdrawerB	directingBdigitalBdesertedBdepictsB	deliciousBdecoration.B
decorationB
courtyard.B	counters.BcornBcookBcontentsBcompleteBcoldBclothes.BcircusBchromeBchoppedBchineseBceiling.B	cathedralBcart.Bcarpet.Bcan.BcageBcabBbread,B	branches.BbootsBblanketsBbikersBbesidesBbelow.BbeakBbathtub,BbarbedBasphaltBarms.Barm.BapronBalmostBairborneBaheadBagedBadvertisementBadorableByellow,BwrapperBwordsBwood.Bwoman,BwavingBwaters.BwatersB
warehouse.BwarBwaiting.B
vegetationBvastB	unloadingBunitedB	turquoiseBtrucks.BtraditionalBtowel.BtowedBtouristsBtoolsBtissueBtile.BtiedB	terminal.Btent.B
telephone.BtaxisBtankerB
tangerinesBswimBsurfsBsuiteBstumpBstudentsBstrawberriesBstrappedBstoolsBstarBstall.Bstage.BspoonsBsoup.BsomewhatBsofa,Bsmile.Bsmall,BslowlyBskylineBski'sBsill.B
silhouetteBshower,B	shoulder.BshoulderB
shoreline.Bserve.B	selectionBseatingBseagullsB	scissors,Bsale.BsailsBrustyBroyalBroundingBroughBremovingBracksBrace.BquicklyBpowderedBpotteryBportraitB	policemenBpointedBpluggedBplainBpizzas.BpilesBperchesBpeppers,BpeacockBpawsBparasailingBparallelBpans.BpackageBownBoutfitsBonions.Bon,BoddBoBnoodlesBnewlyBneck.BnapBnamesBmuseum.Bmound.B	motorbikeBmotelBmostBmonumentBmoneyBmilkBmicrophone.BmexicanBmeetingBmeat.Bmeadow.BmassiveBmainBmacBloungeBlimeB	lighting.BlibraryBledBlaundryB	language.BlanesB	landscapeBjeepBisland.BicingBhutBhospitalBhidingBheads.BhappilyBgrill.BgreensBgarage.BfullyBfriendBforwardBforceBfoods.Bfood,Bflowers,BflourB	flooring.BfixturesB	fireworksBfeet.BfallsBfadedB	extremelyBexhibitBexcitedBenteringBengagedBdualBdrivewayBdriversBdrivenBdriedBdresser.BdollBdesertsBdepotBdeerBdarkenedBdairyBdadB	cupboardsBcrouchedBcribBcrestBcream.BcraneB	courtyardBcountry.BcopBcooksBcontrolsBconeBcompetitionB
comforter.BcomeBcoachBclouds.BcleanedB	classroomBchainB
cellphonesBcatchersBcarvedBcarryBcard.BcansBcamera,Bcage.BcabinB	business.Bbuses.BbunsBblondBblender.BbitBbirds.Bbird.BbinsBbikiniBbikes.BbikerBbats.B	bathroom,BbarefootB	balancingBbakingBbaby.Baway.BanyBairlineBafricanBaffixedBadjacentB4ByouthBwrapper.BwornBwindyBwindingB
wildernessBwild.BwhenB
wheelchairBweather.BwashingBwashBwarmBwalking.B
volleyballBvillageB	vehicles.Bveggies.Bvan.BuniqueBuniform.BuncookedBunableB	umbrella,BtugBtuckedBtropicalBtrainingBtrainerBtouchBtoasterBtimesBtightBtigerB
themselvesBtextBterrain.B	teenagersBtalking.BtabletBtables,Btablecloth.BtB
sweatshirtBsunlitBsuit,B	stretchedBstreetlightBstopsB	stickers.BstatuesBstartBstands.BstB
sprinkles.BspottedBsplitBspeedingBspeedBspeakingBsnacksBsmileyBslidesBslices.BsledBskiiersBsinks,BshapesBshadedBset.BserviceBservesBsee.Bsea.Bscooter.Bscene.BscaleBrubbingBromanBrimBribbonBresortBrelaxBrefrigeratorsBred.BraisingB	racquets.BputsBpushedBpullBpuddleBproductsB	processorBpresentBpoweredBpot.BpocketBplayfulBplain.BpirateBpier.BpickleBpianoBpenguinsBpeersBpedestrians.BparrotsBparentsBpandaBoverpassBoncomingBobject.Bnoodles,B
nightstandBnet.BnearlyBnapkinBmusicBmudBmountainousBmotorcyclistBmomBmixerBmascotBlunch.BloveBlogBloafB	listeningBlingerieBlighthouse.Blift.BliftBlettuce,Bleash.BlearningBlaughingBlabBknife,Bketchup.BiphoneBinternationalBinfantBhornsBhoodieBhitterBhindBhelpBhelmet.B
headboard.BhaulingBhabitat.BguestsBgroceryBgrazesBgrabbingBgrabBgogglesBgatedB	fountain.BfountainBformBforestedBfoliage.BfocusedB	flamingosBflag.B	firetruckBfinishBferryBfeedBfallenBeyes.BexitingB
everythingBevenB
equestrianBelevatedBeightBeat.BdustyBdressingBdressesBdresserBdragonBdoughBdoors.BdoneBdollsBdiskBdishes.Bdishes,Bdisc.BdimB	difficultBdevice.Bdesert.Bdecorations.BcushionBcurvyBcrowdsBcrossesBcreditBcourseBcostumesBcookieBconversationBcontemporaryBconsoleBcondiments.BcompactBcomesBcollarBcobblestoneBcoatsBclock,BcleaningBclean.BchinaB
children'sBchestBcherryBcherriesBchefsBchecksBcheckBcellphones.Bcatcher.BcasesBcartoonBcarrot.BcanopyBcamperBcabinet,Bbush.Bbus,BbundleBbullsBbulletBbuffetBbritishBboxingBbowlingBboogieBblurredBblowBblockingB	blankets.BbestBbeds.BbecauseBbeardedBbeans,BbbqBbannerBballsBbalanceB	backyard.BbackyardBbabiesBaudienceBattemptsBathleticBarrowBarena.BarchwayBapple.BapartBanother.BangleBamBadornedBzebra'sByorkBworksBwolfBwithinBwilderness.BwheelerBwell.B	waterway.B
watermelonBwalledBwall,BwakeBwadesBviaBvendorBvendingBvegetation.Bvases.BvaryingB	varietiesBvansBvacantBuprightBunit.B
unfinishedBunderneath.BtuxedoBturnsBtulipsBtrioBtriangleBtrays.B	transportBtoys.BtotBtortillaBtooBtidyBtideBthoseBthingBtastyBtapedBtallerBtagBsystem.B	supplies.BstripeBstringsBstring.BstreamB
strawberryBstormyBstormBsticks.Bstatue.BstartingBstanceBstadium.BstadiumBstackBstable.BsquirrelBsquatBsquare.Bspot.BspicesBspiceBspectators.BspecialBspeakersBsort.B
somewhere.BsomebodyBsoda.BsoapBsnowsuitBsnowboards.Bsnowboarding.Bsmoke.BslowBsleekBskatesBsimilarBsilverware.BsignalsBsignal.B	shorelineBshelves.BshadowBshade.BsevenBseriousBscoutBscootersBsauce,BsantaBsandwiches.BrvBrunnerBrubberBrouteBropesBroom,Broll.BroastBroadsideBrippedB	restroom.Bresort.Bremote,B
remodeled.B	remodeledB
reflectingB	reflectedBreading.BrawBramps.BprinterBpriceBpreyBpolishedBpointBplushBplayer.BplayedBplates,BplatedBplant,Bplanes.BplainsBpitchersBpipesBpinnedBpinBpieces.BpicksBphotographerBpencilBpeanutB	patternedBpapers.B	overlooksBover.BouthouseBothers.Borange,B
onlookers.Bonions,Bonion.Bolives,Bold-fashionedBnose.BnineBneighborhoodBneedsBnecksBnapkin.BnakedBnBmustacheBmugsBmuffinBmud.Bmouse,BmixtureBmirrorsBmidstBmiddle.Bmen,BmeetBmeanBmealsBmayBmarinaBman'sBmamaBmallBmagazineBlondon.Blodge.Blines.BliftsBliftingBliftedBlicksBlevelBlettingBlegs.Bleft.BlateBlandedBland.BkrispyBknowBknittedBkneelsBkingBkicksBkayakBjapaneseBingredients.BincomingBinclineBilluminatedBhuskyBhungBhugBhookedBhoodedBholder.Bhit.B
historicalBhillsideBhidesBherbsBhelmetsB
helicopterBheavilyBheater.Bhats,B
hamburger.Bgym.Bground,Bgreen.Bgrazing.Bgravy,BgrassesBgranolaBgourmetBgoodsB	giraffes.B	giraffe'sB	gentlemenBgeeseBfruits.BfrizbeeBfreewayBforward.Bflying.BfloweredBfloatsBflipsB
fireplace,Bfire.BfinishedBfingersBfilingBfigurineBfemalesBfeeder.BfeaturedBfacedBeye.BextendedBexaminesBentreeB	entrance.BenormousBenginesBengine.Bend.B	embracingBeggs,BdunkinBduffleBdrawingsB
doughnuts.Bdon'tBdogs,BdivingBdistantBdishwasher,BdirectionalBdilapidatedBdicedBdessert.Bdesign.BdenseBdenB
deliveringBdecorBdeckedBdancingB	dalmatianB	cupcakes.BcubsBcrouchesBcrossed.BcrossedBcrashingBcover.Bcourt,Bcounter,Bcouches.BcoolBcontactB	concrete.B	competingBcompanyBcomfortablyBcoinB	clothing.BclosetBcliffBclaimBcity,BcitiesBcheesyB
cheesecakeBcemeteryBcautionBcars,BcarrierBcanalBcalicoB	cafeteriaB	cabinetryBbutter.BbutcherB
businessesBbunkBbulldogBbuggyBbucket.BbronzeBbreakingBbookcaseBbiscuitsBbiscuitBbinB	bicyclistB	beverage.BberriesBbellsBbattersBbatsBbarrenBbarbecueBbandBbanana,BbalconyBbakeryBbacon,BawaitingBartisticBarrivingBarchB
approachesBappliances,B
apartment.BanchoredBanalogBamongstBamidB
ambulancesBalteredBalone.BallowedBalley.BalarmBairfieldBair,BaccompaniedBaccessories.BaccessoriesBzebras.ByieldBworkstationBwordBwirelessBwipingBwinter.B
windshieldBwindow,BwillBwhitBwheeledBwheel.BweedsBwear.BwaterwayBwatchedBwantsBvestsBvendorsBveggieBvanillaBvalley.Bused.Burinals.BunknownBunitBundergroundBtwentyBtusksBtunnel.BtunnelBtubesBtryBtruck,Btrough.BtripleBtrip.Btricks.BtreatsB	travelersBtrains.Btrain,BtrailersBtoyotaBtower,BtouristBtops.Btop,BtoothbrushesBtookBtimes.BtightlyBthruBthrownB
throughoutBthrewBthoughBthirdBthings.Bthere.BthatsBtendingBtelevisionsBteenageB
tableclothB	surroundsBsupportBsugarBsuburbanB	submergedBstyledBstuntsBstump.BstudioBstrungBstonesBstockedBstewBstepBsteeple.BstatesBstalkBstairwayB
staircase.BsquashB	sprinklesBspreadsBsprayBsprawledBsportBsplashBsparseBsoup,BsolarBsoftB
snowmobileBsnow-coveredBsneakersBsnack.B
smartphoneBslippersBslice.Bsleigh.B	sleeping,BslabBskyscrapersBskinnyBskilletBskeletonB
skatepark.Bskateboards.BsituatedBsitting.BsiteBsingingB
silverwareBsidedBsidecar.BshuttersBshrubs.BshrimpBshotsBsheet.BshareBsetupB	sectionalBseats.BseasonedB
scratchingBsavannahBsausage,Bsandwiches,BsanBsalutingB	sailboatsBsailboatBsafewayBsafetyBsBrun.BrottenBroses.Brope.Brolls.BrollerbladingBrollerBrodBringsBride.B	returningB
residence.BremainsBregularBrefrigerators.BrecreationalBreclinerBrayBraisesBrainingBrailsBradioBracquetsBracks.BpushesB
protrudingBprocessBproBprintsBpretendBpots,BpostsBportableBport.BpoloB	policemanBpoisedBpoemBplaza.BplazaBplaying.Bplay.Bplanter.BplanBplains.BplacingB
pineapplesB
pineapple,BpilotBpillarB	pictured.BpicklesBphotographsBphotographedBphotograph.BperchBpensBpenguinBpeeringBpearsBpeakingBpatronsBparked.B
parachutesBpanelingBpalaceB	painting.BpackingB
overturnedBopensBonesBolivesBoccasionBoakBnutsBnumeralsBnowBnotesBnoodles.BnonBneutralBneighborhood.BneatBnativeBnappingBnapkinsBmustard.BmusicalB
mushrooms,BmurkyBmp3BmovieBmovedBmorningB	monitors.Bmirrors.BmillB
microwavesBmeters.BmessBmediumBmasonBmarbledBmall.BmakerBmagnets.Bmade.Blogs.Blog.BlockedB	location.BlocationBlocalBloading.BlitteredBletBleopardBlemonsBlemonBleggedBleg.BleaveBleafBlayeredBlaptops,BlapseBlane.Blanding.Blamp,BlabeledBknickBkneeBkiwiBkissingBjungle.BjockeysB	jetlinersBjellyBjeanBjamBjacketsBjacket.Bjacket,Bitems,Bitem.BipodBinformationBindicateBimage.BicedBhydrantsBhut.BhumanBhotdogs,Bhorses,Bhorse'sBhornBhoodBhoneyBholidayBholderBhipsterBhillyBhikingBhiddenBhere.BherdingB	headboardBhats.Bharbor.BhandicapBhammockBhammerBhabitatBgymBguitarsBguitarBguideBgroupingBgrillingBgreen,Bgoal.BgoalBglowingBgloomyBglidingBglidesBglasses,BgiveBgiganticBgeishaBgazelleBgame,B
furniture,B	frisbees.BfrisbeBfriends.Bframe.Bforeground,Bfoil.BfoggyBflowingBflooringBfixture.BfixBfilthyBfillBfiledB	figurinesBfiguresB	festival.BfestivalBfedBfarmersBfamiliesBfair.Bfactory.BfactoryBevening.BeuropeanBenvironmentBenglishB
enclosure,B	employeesBelseB	elephant,Beggs.Bedge.BdvdBdusk.BduskBdumpBdryerB	driveway.BdrapedBdozenBdown,BdotBdormBdividedBdirections.BdinerBdiamondBdevicesBdepot.B
departmentBdenimBdeliverB
decoratingBdeckeredBdeck.B	dashboardBcustomerB	curtains.Bcups,BcupboardBcup,BcruisesBcruiseB
crosswalk.B	crossing.Bcross-countryBcrackersBcouldBcostume.BcoolingBcooked.Bcontainers.Bconstruction.Bconsole.B
conferenceB	conductorB
condimentsB
completelyBcompeteBcomboBcombingBcokeBclutterBclosed.Bclocks.B
clipboard.Bcliff.BclassBcladB
cigarette.B
chocolate.BchinBchicken,BchicagoBcheckedBchaseBcharterB
charactersBcerealBcenteredBcelebrationB	celebrateBcatcher,Bcards.BcardsBcanoeBcandle.BcampingBcalvesBcafeBcactusBcableBcabin.Bcabbage,BbuttonsBbustedBbushyBbuns.BbundledBbrowsingBbriefBbridleBbridgesBbricksBbrandBboy,BboxedBbottom.Bbottles.Bbottles,BbottledB	bookcase.BbodiesBboats.BboardedBblockedB
bleachers.Bblack-and-whiteBbiggerBbeyondB
beverages.BbendB	beginningBbeggingBbeds,BbayBbatter.BbasinBbarsBbarrelsBbarrelBballs.Bbalcony.BbalancedBbagelsBbaby'sBavenueB	audience.B	assistingB	ascendingBartworkBarrowsB	arrangingB	appointedBapple,BangryBairwaysBadornBadmiringBadBaccents.B66B5ByourB	youngsterByoungerByogurt.ByogurtByellow.ByaksByachtBwtihBwristBwrappingBwovenBworkedBwork.BwinesBwindsurfingBwindowsill.B
windowsillBwind.B	weatheredBwaxingBwavyBwaving.Bwatermelon.Bwatch,BwashroomB	wanderingBwalls,B	wallpaperBwalking,Bwalk-inBwakingBvisitingBvillage.B	viewpointBvictory.B
verticallyBverticalBveggies,Bvanity,BvalleyBuserBupcloseBunripeBunloadB	underpassBuncleanBumpire.B
umbrellas,BultimateBtyingBtuskedBturtleBturnipsB
tupperwareBtruck'sBtrim.BtrimBtreyBtrekkingBtree,Btrain'sBtrailingBtowingBtowels.Btowels,B
tournamentBtortoiseshellB	tortillasB
toothpasteBtoothbrush,Btomato.Btomato,Btoilets.B
toiletriesB	together,Btofu,BtoesBtoday.Btoast,BtoastBtipBties.Btie,BticketBtiaraBthere'sBthenBthemselves.BtetheredBtestingBtenBteensBteenBtee.Bteam.BtaxiingBtape,Btangerines,BtakeoffBtaken.BsyrupBsweetBsweater.BswanBsushiB	sunflowerBsummerBsugar.Bsubway.B	substanceB	submarineB	styrofoamBstylizedBstylesBstyle.Bstunt.BstudyingBstudentBstrollerBstripesBstrikingBstrewnB	streamingBstraw.B
straddlingBstores.Bstorefront.B
storefrontBstoredBstool.BstoolBstereoBstems.BsteerBstayingBstatues.BstatingBstateBstainedB	squintingB	squattingBsquatsB	spreadingBspread.BsportingBsplitsB	splashingBspinningBspeed.B	spectatorBsparselyBspacesBsoftballB
snowboard,Bsnow,BsniffB	smotheredBsmiling,BslightlyBslices,BsleepyBsled.Bslaw.BslateBskyline.BskillsBskiers.B	skateparkBskateboarding.Bsizes.BsilBsighBshrineBshowroomBshouldBshops.BshopsBshoes,Bshoe.Bshirt,BshipsBship.Bsheets.Bshed.BshearingBshavingBshakeBshaggyBshadyBshadesBsexyBsets.BserverB	separatedBselfB	seeminglyBseeingBsealedB
sculpturesB
sculpture.Bscreens.BscreensB	scooters.BscenicB	savannah.BsamplesBsalmonBsailorsBrundownBrocks,BrocketBrobotBroastingBroadsBroad,BriteBrisingBriseBrings.BringBretroBretailBrest.BrentalBrelish.BrelaxesB	recordingBreadiesBramBrally.Brain,B	railroad.BraftBradiatorBpurchaseBpuncherBprovidesBprovideBprominentlyBpromB	projectorB	progress.BproceedsB
pretendingBpresentsBpreparation.B	practicesBpoundBpotato.BpostcardBpossiblyB	position.B	porcelainB	populatedBpopsicleBpopBpolkaBpole,BpokingBpointyBplowBplayground.B
playgroundBplayers.BplattersBplane,BplacidBpizza'sBpitcher.BpitaBpit.BpiratesBpilotsBpile.Bpie.Bpickup.BpickedBpicBphotographer.BpepsiBpeoplesBpen,BpeakBpcBpawBpatrolBpathwayB	pastries.Bpasta,Bparking.Bparked,Bpark,B	parachuteB	paperworkBpantingBpan,B
pamphlets.BpaleBpackagedBpack.BownersBowner.B
overweightB	overgrownBottomanB	otherwiseBother,BopposingBopen,BonlookerB	olympics.BolympicBolives.BofficialBoctopusBobamaBnuzzlingBnuts.B	nunchuck.BnunBnumbersBnotebookBnormalBnikeB
nighttime.Bnightstands.BnewsBnewerBnetsBneedleBneedBnavyB
navigatingBnastyBnailB
mushrooms.B	mushroom,Bmug.BmuffinsBmountainsideB	mountain,B	motoristsB	motocrossBmoteBmopedsBmoped.BmooredBmoonBmodelsBmittsBminorBmillingBmilkingBmiddle-agedBmetroBmerryBmen'sBmembersBmeatsBmeBmayonnaise.BmaybeB	materialsBmaroonBmarkingsBmarkersBmarker.BmarchBmanualB
mannequinsBmaneuverB	makeshiftBmajorBmaintenanceB	magazinesBmachinesBlowerBlow.BloungingBlounge.BlosBlook.B	longboardBlonelyBlogoBlockBlobbyBliterature.BliteBlit.BlistensBlipBlinkBlinensBlikesBletterBlens.BlensBlemon.BleashesBleapBleanedBleafsBlawBlaughBlaps.BlapsBlanguageB
landscape.BladenBlabradorB
laboratoryBknockedBknivesBkneesBknacksBkitBkissBkeysBkeyBkeepsBkeepBjungleBjokeBjobBithBinvestigatingBinteractBintentlyBintentBintenseBinstructionsBinnB
indicatingB	indicatesBicyBicing.Bice.BhungryBhousingBhotdog.B
horseback.Bhorse-drawnBhornedBhooksBholdersBhitchedBhimself.B	highchairBherdedBheightBheaterBhead,Bhe'sBhazardBhawkBharnessBhandles.BhandheldBhandedBhandbagBhand,BhamBhalvesBhallBhaircutBguidingBguidesBgroveBgroupedBgroomingBgripsBgrinsBgreetingBgreens.Bgravel.BgrasshopperBgrasses.Bgrapes,BgrandBgooseBgoofingBgoods.BgoalieBgnomeBglass,BgentleBgathersBgarbage.BgadgetsBfuzzyBfutonBfurBfunnelB	function.Bfun.BfryingBfruits,BfrisbyBfreezer.BframingBfork,BfontBfoldingBfoamingBflysBfloors,BflippingBflight.Bflavors.B	flatbreadBflatbedBflashBflankedBflags.BfixesBfisheyeBfirehydrantB	finishingBfindBfilmBfillingBfiletBfetaBfencingBfeederBfeathersBfaucet.BfamousBexteriorBexpiredBexoticBexhibit.BexhaustBexceptBexaminedBeuropeBeraBequippedBenjoysBenforcementBenergyB
elephant'sBelectronics.B	elaborateBedgedBeasyBeastBdustBdryingBdryer.B	drinking.BdrawersBdraggingBdozensBdottedBdonuts,BdonkeyBdomeBdoleBdoilyBdockingBdividerBdivesBditchB
disposableB
dishwasherB	disembarkBdiggingBdiagonalBdesks,Bdesigns.B
designatedB
descendingBdentistB
delicious.BdecorateBdecoBdaylightBdanglingBdamagedB	dalmationB	customersBcurvedBcurveBcurtainsBcurryBcurlyBcuriousBcuddlesBcucumberBcuckooBcrust.BcrumbsBcrispBcrewBcreekBcream,BcrateBcrampedBcraftsBcraftB	crackers.Bcows,Bcovers.BcountyBcountrysideBcountertopsBcordBcopsBcoolerBcooked,B	controls.B
controlledBconsumption.BcongratulationsBcondomBconcert.BcompetitiveBcomfortableBcolors.BcoleBcolaB	coca-colaBcocaBcoatedBclubB	cloudlessBcloudBcircus.BciabattaBchoppyBchips.BchargingBcharger.BchannelBchainsB	ceremony.B	cemetery.Bcement.BcellularBcelery.BcelBcauliflower,Bcattle.Bcats.Bcats,Bcat'sBcastBcashBcartonB	carriagesBcarouselBcareBcar'sB	captivityBcaptionBcappedBcapitalBcanopy.BcanoesBcandy.Bcanal.BcamerasBcamelBcallBcafe.BcabsBcabooseBbussBburningBburnersBbundtBbumpingB
buildings,B	building,BbugBbrown,BbrocolliBbringingBbright,Bbridle.B
breakfast.BbraceletBboys.Bboy'sBboxes.Bbox,BbookshelvesB
bookshelf.Bbody.BbmwBblvdBblownBblinds.BblindsBblazerBblack.BbitsB	birthday.B	birdhouseBbiplaneB
bicyclistsBbetterBbench,BbeginsBbeetBbeef,B	bedspreadBbeautifullyBbeatB
batteries,BbatchBbaskets.Bbasin.Bbars.Bbarrier.BbarkBbaleBbalancesBbags.Bbag,BbadBbacon.BbacksideBbackseatBbackedBbackdropBavenue.BaveB	availableBautumnBautomobilesB
automobileB	automaticB	attached.BasiaBartisanBarrive.BarrangementsBarrangement.Baround,BarmyBarenaBareas.BarchedBapproachB	applianceBanimals,BangelesB	amusementBamidstB	amenitiesBalreadyBalmondBalloverBalertB	alcoholicBalaskaB
airplanes.B	airplane,BaidB
afternoon.B	afternoonBadult.B	accident.BaccidentBabout.B95thB65thB33rdB,B"doBzonesBzone.BzombieBzebras,Bzebra.ByummyByouthsByet.ByetB	yellowishBwritesBworld.BworldBwoollyB	wonderfulBwonderBwomen,BwomansBwiredBwings.Bwing.Bwinery.Bwine,Bwindshield.BwindsailBwindsB	windmillsBwildflowers.BwildebeestsBwiimoteBwii,BwifeBwhomBwhiskeyBwhippedBwhere.BwheelieBwheelchair.BwesternBwentB	well-usedBwell-litBweeds.BwedBwayland,Bwaterfront.B
waterfrontB
waterfall.B	waterfallBwater'sB	washroom.BwashedBwarning.BwarmlyB	warehouseB
wallpaper.Bwallet,BwalkedBwalingBwaistBwaggingBwBvisitorBvisible.Bvines.BviewsBviewer.Bvideo.B	victorianBvibrantBvests.Bvest.Bvest,BventilationBvelvetB	vehicularB
vegetarianBvaultedBvase,B
variationsBvanity.BvaneBvandalized.B
vandalizedButensilBusuallyBusualBupwardsBupwardBupstairsBupperBuntidyB
unloading.BunloadedB	uniforms.BunidentifiedBuniBundone.BundevelopedB
underwaterB	undersideB	uncoveredBukraineBuglyBu.s.B
typewriterBtwigsBtwig.BtwelveBtvsBtv,BtundraBtunaBtrunks.Btrunk,Btrucks,BtruckedBtroughBtroubleBtrottingBtrooperBtrollyBtriumphBtrippingBtrimmingBtreedBtree'sBtreat.BtraveledB	trashcansBtransportingBtransportation.BtramBtownsBtowersBtossB	tortilla,BtornB	toppings,Btopping.Btoothpaste,Btoothbrushes.Btool.BtoneBtoliet.BtoiletteBtoilets,BtoilerBtoastingBtiredBtire.BtippedBtinfoil.BtimerBtiltingBtiltedBtilingBtiles.Btiles,BtierBtide.BthumbBthrow.BthreBthinkBthermosBtheresBthem,BthatchedBtextingBtendedBtelevisions.B	telescopeBteenagedBteapotBtealBtea.BtattooedBtatteredBtasting.B
tastefullyBtarBtall,BtakeoutBtakeoff.Btake.Btail.BtaggingBtacosB	tabletop.Bt-ball.BsymbolB	swimming.BswiftlyBsweetsBsweatBswBsurroundB	surprisedBsurfboardingB	surfaces.Bsurface,BsureB	sunshine.BsunshineBsunlightBsunglasses,Bsuits.B
suitcases,BsuchB
successfulBstupidBstuff.B	studying.BstrongB	strollingBstrips.BstripsBstrippedBstrikesBstrikeB
streetcar.Bstreet"BstrapsBstorm.BstoriesBstoreyBstorage.B
stoplight.Bstools.Bstone.BstocksBstittingBstirBstick.BstemBsteamedBstealBsteak,BstationsBstationery,BstaplesB	standing.BstandardBstance.Bstall,BstalksB	staircaseBstainingBstagesBstagedBstacks.BstacksBsqueezedBspots.BspotBsport.Bspoons.Bspoon,BspokeBsplashedBspiresBspire.Bspices.B	specialtyBspearsBspeaksBspeakBspatula.BspareBspanningBspanishB
spaghetti.BspaBsourBsoundBsomeone.BsombreroBsolitaryBsold.BsofasBsodasBsocialBsoarsBsoaringBsoakingBsoakedB	snugglingBsnowing.BsnoozingBsmoothBsmackBslope,BslightBsleighBsleddingBslawBslatsBslab.BskyingBsky,BskullBskis,Bskirt.Bskills.B
skillfullyBskier.Bskating.BskateboardedBskateboard,Bsitting,Bsits.BsippingB
simulator.BsillyBsill,BsilkBsightseeingBsiding.B	sidewalk,BsidecarBside,BsiameseBshrimp,B	shreddingBshowing.B	shouldersBshorts.BshorterBshoppersBshineBshieldsBshelvingBshelves,Bshelter.Bshelf'sBsheeringBshavedBsharpBshape.BshanksBshadowsBshackBset,Bservice.B	seriouslyBsepiaBseperateB
separatingBsellBselfies.BselfiesBseen.BseemBseeds.BseedBsecurityB	sectionedBsecludedBseattleB	seasoningBseashoreB	searchingBsealBsculptures.Bscratch.BscrapsB	scrambledBscottBscoopBscissorBsceneryBscarvesBsayBsaveBsavanna.BsavannaBsaucesBsaucer.BsamsungB	samsoniteBsamplingBsampleBsame.BsalesBsale,Bsails.B	sailboat.Bsafari.BsafariBsaddledBsaddleBsadBrunway,Brunner.BrugbyBrug,BrueBrubyBrowingBrottingBrose.BroostingBroosterBroomsBrooftopBrod.BrockingBrock,BroastedB	roadside.B	riverboatBriver,BrisesBripeningBrink.Briding.Bridge.Brider.Bribbons.BribbonsBrhinoB	revealingBreturnsBreturnedB	retrieverBrestaurant,B	respectedBreservationB
resemblingBresembleBremovesBremodelB	releasingBreignsBregisterBreflectsB
reflectiveBreflection.B
recreationB	recorder.BrecipeB
reception.BrealBread.BratherBraquet.BraquetBrange,BramsBrallyBrails.BrachaelB
racetrack.BracesBracersBquiteBquiltBquicheBqueenBquartersBqBpurse,BpurposeB	purchase.BpumpsBpulled.Bpuddle.Bpublic.BprovidedBprotest.BproppingB	propellorB
propellersB
projector.B
projectionBprogramBprofessionalsBproduce,B	processedBprintedBprincessBpriestBpressingB	preserve.B
presentingB	presentedBpresent.BpreparationBpots.B	potatoes.B	positionsBpose.B	portrait.BportionBportBpopularBpony.BpolishBpoles,BplowingBplayers,Bplastic.Bplants,B	plantainsBplankBpizzeriaBpizzas,B	pitcher'sBpitchedBpitBpirate'sB	pineappleBpin.Bpillows,Bpillow,Bpillar.BpikeBpigeons.Bpickles.Bpickles,Bpick-upBphotoshoppedBphotos.Bphotos,BphotographyBphotographs.Bphoto,Bpetted.BpetsBpestoBperspectiveB	personnelBperformBperfectBpeppers.B
pepperoni,Bpepper.Bpepper,BpeopeBpens,Bpenguin.Bpencils,BpeeksBpedestrians,BpeckingBpeasBpear,BpeakedBpeachB
peacefullyBpeacefulBpeaceBpayBpaws.BpavingBpaul'sBpattiesBpatternsBpatrons.B	pastries,BpastedBpasteBpasta.Bpast.BpassedBpartially-eatenBparlorBparksBparisBparentB	parchmentBpara-sailing
??
Const_5Const*
_output_shapes	
:?'*
dtype0	*̸
value??B??	?'"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_97387
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_97392
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?

Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?	
value?	B?	 B?	
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures*
;

	keras_api
_lookup_layer
_adapt_function*
* 
* 
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 

serving_default* 
* 
7
	keras_api
lookup_table
token_counts*

trace_0* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
R
_initializer
 _create_resource
!_initialize
"_destroy_resource* 
?
#_create_resource
$_initialize
%_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 

&trace_0* 

'trace_0* 

(trace_0* 

)trace_0* 

*trace_0* 

+trace_0* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_4
hash_tableConstConst_1Const_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_97114
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_97429
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameMutableHashTable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_97442??
?
?
__inference_save_fn_97371
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_96971

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?a
?
 __inference__wrapped_model_96837
input_4\
Xsequential_6_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle]
Ysequential_6_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	9
5sequential_6_text_vectorization_string_lookup_equal_y<
8sequential_6_text_vectorization_string_lookup_selectv2_t	
identity	??Ksequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
+sequential_6/text_vectorization/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
2sequential_6/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_6/text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
'sequential_6/text_vectorization/SqueezeSqueeze;sequential_6/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????r
1sequential_6/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
9sequential_6/text_vectorization/StringSplit/StringSplitV2StringSplitV20sequential_6/text_vectorization/Squeeze:output:0:sequential_6/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
?sequential_6/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Asequential_6/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Asequential_6/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
9sequential_6/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_6/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_6/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_6/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_6/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Asequential_6/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_6/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_6/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_6/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_6/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_6/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_6/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_6/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
bsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_6/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
dsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_6/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
lsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
lsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
ksequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
psequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
ksequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
jsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
lsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
jsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
jsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
nsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
osequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountfsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
isequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
msequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
isequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_6/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ksequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_6_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleBsequential_6/text_vectorization/StringSplit/StringSplitV2:values:0Ysequential_6_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
3sequential_6/text_vectorization/string_lookup/EqualEqualBsequential_6/text_vectorization/StringSplit/StringSplitV2:values:05sequential_6_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
6sequential_6/text_vectorization/string_lookup/SelectV2SelectV27sequential_6/text_vectorization/string_lookup/Equal:z:08sequential_6_text_vectorization_string_lookup_selectv2_tTsequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
6sequential_6/text_vectorization/string_lookup/IdentityIdentity?sequential_6/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????~
<sequential_6/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4sequential_6/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
Csequential_6/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_6/text_vectorization/RaggedToTensor/Const:output:0?sequential_6/text_vectorization/string_lookup/Identity:output:0Esequential_6/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_6/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_6/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentityLsequential_6/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOpL^sequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
Ksequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Ksequential_6/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_973878
4key_value_init60145_lookuptableimportv2_table_handle0
,key_value_init60145_lookuptableimportv2_keys2
.key_value_init60145_lookuptableimportv2_values	
identity??'key_value_init60145/LookupTableImportV2?
'key_value_init60145/LookupTableImportV2LookupTableImportV24key_value_init60145_lookuptableimportv2_table_handle,key_value_init60145_lookuptableimportv2_keys.key_value_init60145_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init60145/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?':?'2R
'key_value_init60145/LookupTableImportV2'key_value_init60145/LookupTableImportV2:!

_output_shapes	
:?':!

_output_shapes	
:?'
?
*
__inference_<lambda>_97392
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_97324
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name60146*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
,
__inference__destroyer_97337
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_restore_fn_97379
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
,__inference_sequential_6_layer_call_fn_96995
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_96971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97319

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_signature_wrapper_97114
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_96837o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_97352
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97099
input_4O
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_973328
4key_value_init60145_lookuptableimportv2_table_handle0
,key_value_init60145_lookuptableimportv2_keys2
.key_value_init60145_lookuptableimportv2_values	
identity??'key_value_init60145/LookupTableImportV2?
'key_value_init60145/LookupTableImportV2LookupTableImportV24key_value_init60145_lookuptableimportv2_table_handle,key_value_init60145_lookuptableimportv2_keys.key_value_init60145_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init60145/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?':?'2R
'key_value_init60145/LookupTableImportV2'key_value_init60145/LookupTableImportV2:!

_output_shapes	
:?':!

_output_shapes	
:?'
?
?
__inference__traced_save_97429
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1savev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
: 
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97047
input_4O
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_6_layer_call_fn_96904
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_96893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?n
?
__inference_adapt_step_97189
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes
: *
output_shapes
: *
output_types
2P
StringLowerStringLowerIteratorGetNext:components:0*
_output_shapes
: ?
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*
_output_shapes
: *2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite d
StringSplit/stackPackStaticRegexReplace:output:0*
N*
T0*
_output_shapes
:^
StringSplit/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
%StringSplit/StringSplit/StringSplitV2StringSplitV2StringSplit/stack:output:0&StringSplit/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:|
+StringSplit/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ~
-StringSplit/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ~
-StringSplit/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
%StringSplit/StringSplit/strided_sliceStridedSlice/StringSplit/StringSplit/StringSplitV2:indices:04StringSplit/StringSplit/strided_slice/stack:output:06StringSplit/StringSplit/strided_slice/stack_1:output:06StringSplit/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskw
-StringSplit/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/StringSplit/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/StringSplit/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'StringSplit/StringSplit/strided_slice_1StridedSlice-StringSplit/StringSplit/StringSplitV2:shape:06StringSplit/StringSplit/strided_slice_1/stack:output:08StringSplit/StringSplit/strided_slice_1/stack_1:output:08StringSplit/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
NStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast.StringSplit/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
PStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast0StringSplit/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
XStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeRStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
XStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
WStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdaStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0aStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
\StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater`StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0eStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
WStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast^StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
VStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxRStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0cStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
XStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
VStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2_StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0aStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
VStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul[StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumTStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumTStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ZStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
[StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountRStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0^StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0cStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
UStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
PStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumbStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0^StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
YStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
UStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
PStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2bStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0VStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0^StringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????w
-StringSplit/RaggedGetItem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
/StringSplit/RaggedGetItem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/StringSplit/RaggedGetItem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'StringSplit/RaggedGetItem/strided_sliceStridedSliceYStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06StringSplit/RaggedGetItem/strided_slice/stack:output:08StringSplit/RaggedGetItem/strided_slice/stack_1:output:08StringSplit/RaggedGetItem/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_masky
/StringSplit/RaggedGetItem/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1StringSplit/RaggedGetItem/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1StringSplit/RaggedGetItem/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)StringSplit/RaggedGetItem/strided_slice_1StridedSliceYStringSplit/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:08StringSplit/RaggedGetItem/strided_slice_1/stack:output:0:StringSplit/RaggedGetItem/strided_slice_1/stack_1:output:0:StringSplit/RaggedGetItem/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_masky
/StringSplit/RaggedGetItem/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1StringSplit/RaggedGetItem/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1StringSplit/RaggedGetItem/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)StringSplit/RaggedGetItem/strided_slice_2StridedSlice0StringSplit/RaggedGetItem/strided_slice:output:08StringSplit/RaggedGetItem/strided_slice_2/stack:output:0:StringSplit/RaggedGetItem/strided_slice_2/stack_1:output:0:StringSplit/RaggedGetItem/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_masky
/StringSplit/RaggedGetItem/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1StringSplit/RaggedGetItem/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1StringSplit/RaggedGetItem/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)StringSplit/RaggedGetItem/strided_slice_3StridedSlice2StringSplit/RaggedGetItem/strided_slice_1:output:08StringSplit/RaggedGetItem/strided_slice_3/stack:output:0:StringSplit/RaggedGetItem/strided_slice_3/stack_1:output:0:StringSplit/RaggedGetItem/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maska
StringSplit/RaggedGetItem/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R?
/StringSplit/RaggedGetItem/strided_slice_4/stackPack2StringSplit/RaggedGetItem/strided_slice_2:output:0*
N*
T0	*
_output_shapes
:?
1StringSplit/RaggedGetItem/strided_slice_4/stack_1Pack2StringSplit/RaggedGetItem/strided_slice_3:output:0*
N*
T0	*
_output_shapes
:?
1StringSplit/RaggedGetItem/strided_slice_4/stack_2Pack(StringSplit/RaggedGetItem/Const:output:0*
N*
T0	*
_output_shapes
:?
)StringSplit/RaggedGetItem/strided_slice_4StridedSlice.StringSplit/StringSplit/StringSplitV2:values:08StringSplit/RaggedGetItem/strided_slice_4/stack:output:0:StringSplit/RaggedGetItem/strided_slice_4/stack_1:output:0:StringSplit/RaggedGetItem/strided_slice_4/stack_2:output:0*
Index0	*
T0*#
_output_shapes
:?????????r
/StringSplit/RaggedGetItem/strided_slice_5/ConstConst*
_output_shapes
: *
dtype0*
valueB ?
)StringSplit/RaggedGetItem/strided_slice_5StridedSlice2StringSplit/RaggedGetItem/strided_slice_4:output:08StringSplit/RaggedGetItem/strided_slice_5/Const:output:08StringSplit/RaggedGetItem/strided_slice_5/Const:output:08StringSplit/RaggedGetItem/strided_slice_5/Const:output:0*
Index0*
T0*#
_output_shapes
:?????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?

ExpandDims
ExpandDims2StringSplit/RaggedGetItem/strided_slice_5:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
ReshapeReshapeExpandDims:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?
F
__inference__creator_97342
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_9*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
!__inference__traced_restore_97442
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: 

identity_1??2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
IdentityIdentityfile_prefix3^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: S

Identity_1IdentityIdentity:output:0^NoOp_1*
T0*
_output_shapes
: }
NoOp_1NoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_96893

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_6_layer_call_fn_97202

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_96893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97267

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*2
pattern'%!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????2       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????2*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:?????????2?
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_6_layer_call_fn_97215

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_96971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
.
__inference__initializer_97347
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_40
serving_default_input_4:0?????????H
text_vectorization2
StatefulPartitionedCall_1:0	?????????2tensorflow/serving/predict:?U
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures"
_tf_keras_sequential
P

	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_1
trace_2
trace_32?
,__inference_sequential_6_layer_call_fn_96904
,__inference_sequential_6_layer_call_fn_97202
,__inference_sequential_6_layer_call_fn_97215
,__inference_sequential_6_layer_call_fn_96995?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ztrace_0ztrace_1ztrace_2ztrace_3
?
trace_0
trace_1
trace_2
trace_32?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97267
G__inference_sequential_6_layer_call_and_return_conditional_losses_97319
G__inference_sequential_6_layer_call_and_return_conditional_losses_97047
G__inference_sequential_6_layer_call_and_return_conditional_losses_97099?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ztrace_0ztrace_1ztrace_2ztrace_3
?B?
 __inference__wrapped_model_96837input_4"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
serving_default"
signature_map
"
_generic_user_object
L
	keras_api
lookup_table
token_counts"
_tf_keras_layer
?
trace_02?
__inference_adapt_step_97189?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ztrace_0
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_6_layer_call_fn_96904input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_6_layer_call_fn_97202inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_6_layer_call_fn_97215inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_6_layer_call_fn_96995input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97267inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97319inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97047input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97099input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_97114input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
f
_initializer
 _create_resource
!_initialize
"_destroy_resourceR jtf.StaticHashTable
J
#_create_resource
$_initialize
%_destroy_resourceR Z
 ,-
?B?
__inference_adapt_step_97189iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
&trace_02?
__inference__creator_97324?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z&trace_0
?
'trace_02?
__inference__initializer_97332?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z'trace_0
?
(trace_02?
__inference__destroyer_97337?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z(trace_0
?
)trace_02?
__inference__creator_97342?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z)trace_0
?
*trace_02?
__inference__initializer_97347?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z*trace_0
?
+trace_02?
__inference__destroyer_97352?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z+trace_0
?B?
__inference__creator_97324"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_97332"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_97337"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_97342"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_97347"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_97352"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_97371checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_97379restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant6
__inference__creator_97324?

? 
? "? 6
__inference__creator_97342?

? 
? "? 8
__inference__destroyer_97337?

? 
? "? 8
__inference__destroyer_97352?

? 
? "? ?
__inference__initializer_9733223?

? 
? "? :
__inference__initializer_97347?

? 
? "? ?
 __inference__wrapped_model_96837?./00?-
&?#
!?
input_4?????????
? "G?D
B
text_vectorization,?)
text_vectorization?????????2	\
__inference_adapt_step_97189<12?/
(?%
#? ?	
? IteratorSpec 
? "
 y
__inference_restore_fn_97379YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_97371?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97047g./08?5
.?+
!?
input_4?????????
p 

 
? "%?"
?
0?????????2	
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97099g./08?5
.?+
!?
input_4?????????
p

 
? "%?"
?
0?????????2	
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97267f./07?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????2	
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_97319f./07?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????2	
? ?
,__inference_sequential_6_layer_call_fn_96904Z./08?5
.?+
!?
input_4?????????
p 

 
? "??????????2	?
,__inference_sequential_6_layer_call_fn_96995Z./08?5
.?+
!?
input_4?????????
p

 
? "??????????2	?
,__inference_sequential_6_layer_call_fn_97202Y./07?4
-?*
 ?
inputs?????????
p 

 
? "??????????2	?
,__inference_sequential_6_layer_call_fn_97215Y./07?4
-?*
 ?
inputs?????????
p

 
? "??????????2	?
#__inference_signature_wrapper_97114?./0;?8
? 
1?.
,
input_4!?
input_4?????????"G?D
B
text_vectorization,?)
text_vectorization?????????2	