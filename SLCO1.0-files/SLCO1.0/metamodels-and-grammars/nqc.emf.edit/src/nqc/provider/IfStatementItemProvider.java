/**
 */
package nqc.provider;


import java.util.Collection;
import java.util.List;

import nqc.IfStatement;
import nqc.NqcFactory;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ViewerNotification;

/**
 * This is the item provider adapter for a {@link nqc.IfStatement} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class IfStatementItemProvider
	extends ControlStructureItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IfStatementItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

		}
		return itemPropertyDescriptors;
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(NqcPackage.eINSTANCE.getIfStatement_Condition());
			childrenFeatures.add(NqcPackage.eINSTANCE.getIfStatement_Consequence());
			childrenFeatures.add(NqcPackage.eINSTANCE.getIfStatement_Alternative());
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns IfStatement.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/IfStatement"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_IfStatement_type");
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(IfStatement.class)) {
			case NqcPackage.IF_STATEMENT__CONDITION:
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Condition(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createAcquireStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createAssignmentStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBlockStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createBreakStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createContinueStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createDoWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createEmptyStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createForStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createGoToStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createIfStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createMonitorStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createQuaternaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createRepeatStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createReturnStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSenaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createStartStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createStopStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSubroutineCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createSwitchStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createTernaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createUntilStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Consequence(),
				 NqcFactory.eINSTANCE.createWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createAcquireStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createAssignmentStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBlockStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createBreakStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createContinueStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createDoWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createEmptyStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createForStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createGoToStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createIfStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createMonitorStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createQuaternaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createRepeatStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createReturnStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSenaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createStartStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createStopStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSubroutineCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createSwitchStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createTernaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createUntilStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getIfStatement_Alternative(),
				 NqcFactory.eINSTANCE.createWhileStatement()));
	}

	/**
	 * This returns the label text for {@link org.eclipse.emf.edit.command.CreateChildCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getCreateChildText(Object owner, Object feature, Object child, Collection<?> selection) {
		Object childFeature = feature;
		Object childObject = child;

		boolean qualify =
			childFeature == NqcPackage.eINSTANCE.getIfStatement_Condition() ||
			childFeature == NqcPackage.eINSTANCE.getIfStatement_Consequence() ||
			childFeature == NqcPackage.eINSTANCE.getIfStatement_Alternative();

		if (qualify) {
			return getString
				("_UI_CreateChild_text2",
				 new Object[] { getTypeText(childObject), getFeatureText(childFeature), getTypeText(owner) });
		}
		return super.getCreateChildText(owner, feature, child, selection);
	}

}
